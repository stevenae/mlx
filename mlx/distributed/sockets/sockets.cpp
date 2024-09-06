// Copyright © 2024 Apple Inc.

#include <arpa/inet.h>
#include <json.hpp>
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

constexpr const size_t PACKET_SIZE = 262144;
constexpr const int CONN_ATTEMPTS = 5;
constexpr const int CONN_WAIT = 1000;

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

struct address_t {
  sockaddr_storage addr;
  socklen_t len;

  const sockaddr* sockaddr() {
    return (struct sockaddr*)&addr;
  }
};

address_t parse_address(std::string ip, std::string port) {
  struct addrinfo hints, *res;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int status = getaddrinfo(ip.c_str(), port.c_str(), &hints, &res);
  if (status != 0) {
    std::ostringstream msg;
    msg << "Can't parse peer address " << ip << ":" << port;
    throw std::runtime_error(msg.str());
  }

  address_t result;
  memcpy(&result.addr, res->ai_addr, res->ai_addrlen);
  result.len = res->ai_addrlen;
  freeaddrinfo(res);

  return result;
}

std::vector<address_t> load_peers() {
  std::vector<address_t> peers;
  std::ifstream f;

  if (const char* hostfile_buf = std::getenv("MLX_HOSTFILE")) {
    f.open(hostfile_buf);
  } else {
    return peers;
  }

  json hosts = json::parse(f);
  for (auto& h : hosts) {
    peers.push_back(std::move(parse_address(
        h["ip"].template get<std::string>(),
        h["port"].template get<std::string>())));
  }

  return peers;
}

struct GroupImpl {
  GroupImpl(std::vector<address_t> peers, int rank, bool global)
      : rank_(rank), global_(global), pool_(4), sockets_(peers.size(), -1) {
    if (rank_ > 0 && rank_ >= peers.size()) {
      throw std::runtime_error(
          "Rank cannot be larger than the size of the group");
    }

    int success;

    // If we are expecting anyone to connect to us
    if (rank_ + 1 < peers.size()) {
      // Create the socket to wait for connections from the peers
      int sock = socket(AF_INET, SOCK_STREAM, 0);
      if (sock < 0) {
        std::ostringstream msg;
        msg << "Couldn't create socket (error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }

      // Make sure we can launch immediately after shutdown by setting the
      // reuseaddr option so that we don't get address already in use errors
      int enable = 1;
      success =
          setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
      if (success < 0) {
        shutdown(sock, 2);
        close(sock);
        std::ostringstream msg;
        msg << "Couldn't enable reuseaddr (rank: " << rank_
            << " error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
      success =
          setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));
      if (success < 0) {
        shutdown(sock, 2);
        close(sock);
        std::ostringstream msg;
        msg << "Couldn't enable reuseport (rank: " << rank_
            << " error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }

      // Bind it to the port
      success = bind(sock, peers[rank_].sockaddr(), peers[rank_].len);
      if (success < 0) {
        shutdown(sock, 2);
        close(sock);
        std::ostringstream msg;
        msg << "Couldn't bind socket (rank: " << rank_ << " error: " << errno
            << ")";
        throw std::runtime_error(msg.str());
      }

      // Wait for connections
      success = listen(sock, 0);
      if (success < 0) {
        shutdown(sock, 2);
        close(sock);
        std::ostringstream msg;
        msg << "Couldn't listen (error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
      for (int i = 0; i < peers.size() - rank_ - 1; i++) {
        int peer_socket = accept(sock, nullptr, nullptr);
        if (peer_socket < 0) {
          shutdown(sock, 2);
          close(sock);
          std::ostringstream msg;
          msg << "Accept failed (error: " << errno << ")";
          throw std::runtime_error(msg.str());
        }
        sockets_[peers.size() - 1 - i] = peer_socket;
      }

      // Close the listening socket
      shutdown(sock, 2);
      close(sock);
    }

    // Connect to the peers with smaller rank
    for (int i = 0; i < rank_; i++) {
      sockets_[i] = socket(AF_INET, SOCK_STREAM, 0);
      if (sockets_[i] < 0) {
        std::ostringstream msg;
        msg << "Couldn't create socket (error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
      for (int attempt = 0; attempt < CONN_ATTEMPTS; attempt++) {
        if (attempt > 0) {
          int wait = (1 << (attempt - 1)) * CONN_WAIT;
          std::this_thread::sleep_for(std::chrono::milliseconds(wait));
        }
        success = connect(sockets_[i], peers[i].sockaddr(), peers[i].len);
        if (success == 0) {
          break;
        }
      }
      if (success < 0) {
        std::ostringstream msg;
        msg << "Couldn't connect (rank: " << rank_ << " to: " << i
            << " error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
    }
  }

  ~GroupImpl() {
    if (global_) {
      for (int sock : sockets_) {
        shutdown(sock, 2);
        close(sock);
      }
    }
  }

  int rank() {
    return rank_;
  }

  int size() {
    return std::max(sockets_.size(), 1ul);
  }

  void send(const char* buf, size_t len, int dst) {
    while (len > 0) {
      ssize_t r = ::send(sockets_[dst], buf, len, 0);
      if (r <= 0) {
        std::ostringstream msg;
        msg << "Send of " << len << " bytes failed (errno: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
      buf += r;
      len -= r;
    }
  }

  void recv(char* buf, size_t len, int src) {
    while (len > 0) {
      ssize_t r = ::recv(sockets_[src], buf, len, 0);
      if (r <= 0) {
        std::ostringstream msg;
        msg << "Recv of " << len << " bytes failed (errno: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
      buf += r;
      len -= r;
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
  std::vector<int> sockets_;
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
    auto peers = load_peers();
    int rank = 0;
    if (const char* rank_buf = std::getenv("MLX_RANK")) {
      rank = std::atoi(rank_buf);
    }
    if (peers.size() == 0) {
      if (strict) {
        throw std::runtime_error("Can't initialize distributed");
      }
    }
    global_group = std::make_shared<GroupImpl>(std::move(peers), rank, true);
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
