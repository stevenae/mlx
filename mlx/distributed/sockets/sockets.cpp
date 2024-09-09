// Copyright © 2024 Apple Inc.

#include <arpa/inet.h>
#include <json.hpp>
#include <net/ndrv.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "mlx/backend/common/copy.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/io/threadpool.h"

#define SWITCH_TYPE(x, ...)  \
  switch ((x).dtype()) {     \
    case bool_: {            \
      using T = bool;        \
      __VA_ARGS__;           \
    } break;                 \
    case int8: {             \
      using T = int8_t;      \
      __VA_ARGS__;           \
    } break;                 \
    case int16: {            \
      using T = int16_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case int32: {            \
      using T = int32_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case int64: {            \
      using T = int64_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case uint8: {            \
      using T = uint8_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case uint16: {           \
      using T = uint16_t;    \
      __VA_ARGS__;           \
    } break;                 \
    case uint32: {           \
      using T = uint32_t;    \
      __VA_ARGS__;           \
    } break;                 \
    case uint64: {           \
      using T = uint64_t;    \
      __VA_ARGS__;           \
    } break;                 \
    case bfloat16: {         \
      using T = bfloat16_t;  \
      __VA_ARGS__;           \
    } break;                 \
    case float16: {          \
      using T = float16_t;   \
      __VA_ARGS__;           \
    } break;                 \
    case float32: {          \
      using T = float;       \
      __VA_ARGS__;           \
    } break;                 \
    case complex64: {        \
      using T = complex64_t; \
      __VA_ARGS__;           \
    } break;                 \
  }

constexpr const size_t PACKET_SIZE = 1408;
constexpr const uint16_t ETHER_TYPE = 32923;
constexpr const uint16_t ETHER_TYPE_NTOHS = ntohs(ETHER_TYPE);

using json = nlohmann::json;

namespace mlx::core::distributed {

namespace {

template <typename T>
void sum_inplace(const T* input, T* output, size_t N) {
  while (N-- > 0) {
    *output += *input;
    input++;
    output++;
  }
}

void sum_inplace(const array& input, array& output) {
  SWITCH_TYPE(
      input, sum_inplace(input.data<T>(), output.data<T>(), input.size()));
}

array ensure_row_contiguous(const array& arr) {
  if (arr.flags().row_contiguous) {
    return arr;
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy(arr, arr_copy, CopyType::General);
    return arr_copy;
  }
}

struct mac_address {
  uint8_t raw[6] = {0};

  mac_address(const std::string& address) {
    auto hex_to_int = [](const char c) -> uint8_t {
      if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
      }

      if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
      }

      if (c >= '0' && c <= '9') {
        return c - '0';
      }

      return 0;
    };

    int idx = 0;
    int cnt = 0;
    for (const auto c : address) {
      if (c == ':') {
        idx += 1;
        cnt = 0;
        if (idx >= 6) {
          break;
        }
      } else {
        raw[idx] <<= 4 * cnt;
        raw[idx] += hex_to_int(c);
      }
    }
  }

  void to_buffer(char* buf) {
    for (int i = 0; i < 6; i++) {
      buf[i] = ((char*)raw)[i];
    }
  }
};

std::pair<std::string, std::vector<mac_address>> parse_config() {
  std::vector<mac_address> peers;
  std::ifstream f;

  if (const char* hostfile_buf = std::getenv("MLX_HOSTFILE")) {
    f.open(hostfile_buf);
  } else {
    return {"lo0", peers};
  }

  json config = json::parse(f);
  for (auto& h : config["peers"]) {
    peers.emplace_back(h.get<std::string>());
  }

  return {config["interface"].get<std::string>(), peers};
}

struct GroupImpl {
  GroupImpl(
      const std::string& interface,
      std::vector<mac_address> peers,
      int rank,
      bool global)
      : rank_(rank), global_(global), pool_(1), peers_(std::move(peers)) {
    if (rank_ > 0 && rank_ >= peers_.size()) {
      throw std::runtime_error(
          "Rank cannot be larger than the size of the group");
    }

    if (peers_.size() == 0) {
      return;
    }

    // Make the socket
    socket_ = socket(PF_NDRV, SOCK_RAW, 0);
    if (socket_ < 0) {
      std::ostringstream msg;
      msg << "Couldn't create socket (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    // Make the address to bind the socket
    std::copy(interface.begin(), interface.end(), (char*)sockaddr_.snd_name);
    sockaddr_.snd_family = PF_NDRV;
    sockaddr_.snd_len = sizeof(sockaddr_);
    if (bind(socket_, (sockaddr*)&sockaddr_, sizeof(sockaddr_)) < 0) {
      std::ostringstream msg;
      msg << "Couldn't bind socket (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    // Tell the kernel to filter and select for ETHER_TYPE
    ndrv_protocol_desc desc;
    ndrv_demux_desc demux_desc;
    desc.version = NDRV_PROTOCOL_DESC_VERS;
    desc.protocol_family = ETHER_TYPE;
    desc.demux_count = 1;
    desc.demux_list = &demux_desc;
    demux_desc.type = NDRV_DEMUXTYPE_ETHERTYPE;
    demux_desc.length = sizeof(uint16_t);
    demux_desc.data.ether_type = ETHER_TYPE_NTOHS;
    if (setsockopt(
            socket_, SOL_NDRVPROTO, NDRV_SETDMXSPEC, &desc, sizeof(desc)) < 0) {
      std::ostringstream msg;
      msg << "Couldn't set socket option (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }
  }

  ~GroupImpl() {
    if (global_ && socket_ > 0) {
      close(socket_);
    }
  }

  int rank() {
    return rank_;
  }

  int size() {
    return std::max(peers_.size(), 1ul);
  }

  void send_packet(const char* buf, size_t len, int dst) {
    char packet[1500];
    peers_[dst].to_buffer(packet);
    peers_[rank_].to_buffer(packet + sizeof(mac_address));
    memcpy(packet + 2 * sizeof(mac_address), &ETHER_TYPE_NTOHS, sizeof(ETHER_TYPE_NTOHS));
    constexpr int header_len = 2 * sizeof(mac_address) + 2;
    memcpy(packet + header_len, buf, len);
    int r = sendto(
        socket_,
        packet,
        len + header_len,
        0,
        (sockaddr*)&sockaddr_,
        sizeof(sockaddr_));
    if (r < 0) {
      std::ostringstream msg;
      msg << "Send failed (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }
  }

  void recv_packet(char* buf, size_t len, int src) {
    char packet[1500];
    constexpr int header_len = 2 * sizeof(mac_address) + 2;
    int r = ::recv(socket_, packet, len + header_len, 0);
    if (r < 0) {
      std::ostringstream msg;
      msg << "Send failed (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }
    memcpy(buf, packet + header_len, len);
  }

  void send(const char* buf, size_t len, int dst) {
    while (len > 0) {
      size_t l = std::min(len, PACKET_SIZE);
      send_packet(buf, l, dst);
      buf += l;
      len -= l;
    }
  }

  void recv(char* buf, size_t len, int src) {
    while (len > 0) {
      size_t l = std::min(len, PACKET_SIZE);
      recv_packet(buf, l, src);
      buf += l;
      len -= l;
    }
  }

  template <typename T>
  void send_recv_sum(char* buf, size_t len, int peer) {
    char recv_buffer[2 * PACKET_SIZE];
    char* recv_buffers[2];
    recv_buffers[0] = recv_buffer;
    recv_buffers[1] = recv_buffer + PACKET_SIZE;
    std::future<void> sent, received;
    size_t n_blocks = (len + PACKET_SIZE - 1) / PACKET_SIZE;

    for (size_t b = 0; b < n_blocks; b++) {
      if (b > 0) {
        sent.wait();
        received.wait();
      }
      size_t l = std::min(len - b * PACKET_SIZE, PACKET_SIZE);
      if (rank_ < peer) {
        sent = send_async(buf + b * PACKET_SIZE, l, peer);
        received = recv_async(recv_buffers[b % 2], l, peer);
      } else {
        received = recv_async(recv_buffers[b % 2], l, peer);
        sent = send_async(buf + b * PACKET_SIZE, l, peer);
      }
      if (b > 0) {
        sum_inplace(
            (const T*)recv_buffers[(b - 1) % 2],
            (T*)(buf + (b - 1) * PACKET_SIZE),
            PACKET_SIZE / sizeof(T));
      }
    }
    sent.wait();
    received.wait();
    size_t l = std::min(len - (n_blocks - 1) * PACKET_SIZE, PACKET_SIZE);
    sum_inplace(
        (const T*)recv_buffers[(n_blocks - 1) % 2],
        (T*)(buf + (n_blocks - 1) * PACKET_SIZE),
        l / sizeof(T));
  }

  void send_recv_sum(array& out, int peer) {
    SWITCH_TYPE(out, send_recv_sum<T>(out.data<char>(), out.nbytes(), peer));
  }

  std::future<void> send_async(const char* buf, size_t len, int dst) {
    return pool_.enqueue(
        [this, buf, len, dst]() { this->send(buf, len, dst); });
  }

  std::future<void> recv_async(char* buf, size_t len, int src) {
    return pool_.enqueue(
        [this, buf, len, src]() { this->recv(buf, len, src); });
  }

 private:
  int rank_;
  bool global_;
  ThreadPool pool_;
  std::vector<mac_address> peers_;
  sockaddr_ndrv sockaddr_;
  int socket_;
};

} // namespace

bool is_available() {
  return true;
}

int Group::rank() {
  return std::static_pointer_cast<GroupImpl>(group_)->rank();
}

int Group::size() {
  return std::static_pointer_cast<GroupImpl>(group_)->size();
}

Group Group::split(int color, int key) {
  throw std::runtime_error("Splitting not supported yet");
}

Group init(bool strict /* = false */) {
  static std::shared_ptr<GroupImpl> global_group = nullptr;

  if (global_group == nullptr) {
    auto [iface, peers] = parse_config();
    int rank = 0;
    if (const char* rank_buf = std::getenv("MLX_RANK")) {
      rank = std::atoi(rank_buf);
    }
    if (peers.size() == 0) {
      if (strict) {
        throw std::runtime_error("Can't initialize distributed");
      }
    }
    global_group =
        std::make_shared<GroupImpl>(iface, std::move(peers), rank, true);
  }
  return Group(global_group);
}

namespace detail {

Stream communication_stream() {
  static Stream comm_stream = new_stream(Device::cpu);
  return comm_stream;
}

void all_sum(Group group_, const array& input_, array& output) {
  auto group = std::static_pointer_cast<GroupImpl>(group_.raw_group());
  array input = ensure_row_contiguous(input_);

  int size = group->size();
  int rank = group->rank();

  if ((size & (size - 1)) != 0) {
    throw std::runtime_error("Only powers of 2 are currently supported");
  }

  // If not inplace all reduce then copy the input to the output first.
  if (input.data<void>() != output.data<void>()) {
    std::memcpy(output.data<char>(), input.data<char>(), input.nbytes());
  }

  // Butterfly all reduce
  for (int distance = 1; distance <= size / 2; distance *= 2) {
    group->send_recv_sum(output, rank ^ distance);
  }
}

void all_gather(Group group_, const array& input_, array& output) {
  auto group = std::static_pointer_cast<GroupImpl>(group_.raw_group());
  array input = ensure_row_contiguous(input_);
  std::future<void> sent;
  std::future<void> received;

  int rank = group->rank();
  int size = group->size();

  if ((size & (size - 1)) != 0) {
    throw std::runtime_error("Only powers of 2 are currently supported");
  }

  // Butterfly all gather
  int peer = rank ^ 1;
  if (peer < rank) {
    received = group->recv_async(
        output.data<char>() + peer * input.nbytes(), input.nbytes(), peer);
    sent = group->send_async(input.data<char>(), input.nbytes(), peer);
  } else {
    sent = group->send_async(input.data<char>(), input.nbytes(), peer);
    received = group->recv_async(
        output.data<char>() + peer * input.nbytes(), input.nbytes(), peer);
  }
  std::memcpy(
      output.data<char>() + rank * input.nbytes(),
      input.data<char>(),
      input.nbytes());

  for (int distance = 2; distance <= size / 2; distance *= 2) {
    sent.wait();
    received.wait();
    int peer = rank ^ distance;
    int their_offset = peer & ~(distance - 1);
    int our_offset = rank & ~(distance - 1);

    if (peer < rank) {
      received = group->recv_async(
          output.data<char>() + their_offset * input.nbytes(),
          distance * input.nbytes(),
          peer);
      sent = group->send_async(
          output.data<char>() + our_offset * input.nbytes(),
          distance * input.nbytes(),
          peer);
    } else {
      sent = group->send_async(
          output.data<char>() + our_offset * input.nbytes(),
          distance * input.nbytes(),
          peer);
      received = group->recv_async(
          output.data<char>() + their_offset * input.nbytes(),
          distance * input.nbytes(),
          peer);
    }
  }
  sent.wait();
  received.wait();
}

void send(Group group_, const array& input_, int dst) {
  array input = ensure_row_contiguous(input_);
  auto group = std::static_pointer_cast<GroupImpl>(group_.raw_group());
  group->send(input.data<char>(), input.nbytes(), dst);
}

void recv(Group group_, array& out, int src) {
  auto group = std::static_pointer_cast<GroupImpl>(group_.raw_group());
  group->recv(out.data<char>(), out.nbytes(), src);
}

} // namespace detail

} // namespace mlx::core::distributed
