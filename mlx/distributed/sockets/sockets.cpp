// Copyright © 2024 Apple Inc.

#include <arpa/inet.h>
#include <json.hpp>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"

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
  switch (input.dtype()) {
    case bool_:
      return sum_inplace(input.data<bool>(), output.data<bool>(), input.size());
    case int8:
      return sum_inplace(
          input.data<int8_t>(), output.data<int8_t>(), input.size());
    case uint8:
      return sum_inplace(
          input.data<uint8_t>(), output.data<uint8_t>(), input.size());
    case int16:
      return sum_inplace(
          input.data<int16_t>(), output.data<int16_t>(), input.size());
    case uint16:
      return sum_inplace(
          input.data<uint16_t>(), output.data<uint16_t>(), input.size());
    case int32:
      return sum_inplace(
          input.data<int32_t>(), output.data<int32_t>(), input.size());
    case uint32:
      return sum_inplace(
          input.data<uint32_t>(), output.data<uint32_t>(), input.size());
    case int64:
      return sum_inplace(
          input.data<int64_t>(), output.data<int64_t>(), input.size());
    case uint64:
      return sum_inplace(
          input.data<uint64_t>(), output.data<uint64_t>(), input.size());
    case float16:
      return sum_inplace(
          input.data<float16_t>(), output.data<float16_t>(), input.size());
    case bfloat16:
      return sum_inplace(
          input.data<bfloat16_t>(), output.data<bfloat16_t>(), input.size());
    case float32:
      return sum_inplace(
          input.data<float>(), output.data<float>(), input.size());
    case complex64:
      return sum_inplace(
          input.data<complex64_t>(), output.data<complex64_t>(), input.size());
  }
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
  hints.ai_socktype = SOCK_DGRAM;

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
      : peers_(std::move(peers)), rank_(rank), global_(global) {
    if (rank_ > 0 && rank_ >= peers_.size()) {
      throw std::runtime_error(
          "Rank cannot be larger than the size of the group");
    }
    if (global_ && rank_ < peers_.size()) {
      socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
      if (socket_fd_ < 0) {
        std::ostringstream msg;
        msg << "Couldn't create socket (error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
      int success =
          bind(socket_fd_, peers_[rank_].sockaddr(), peers_[rank_].len);
      if (success < 0) {
        std::ostringstream msg;
        msg << "Couldn't bind socket (error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
    }
  }
  ~GroupImpl() {
    if (global_) {
      close(socket_fd_);
    }
  }

  int rank() {
    return rank_;
  }

  int size() {
    return std::max(peers_.size(), 1ul);
  }

  void send(const char* buf, size_t len, int dst) {
    ssize_t r = sendto(
        socket_fd_, buf, len, 0, peers_[dst].sockaddr(), peers_[dst].len);
    if (r < 0) {
      throw std::runtime_error("Send failed.");
    }
  }

  void recv(char* buf, size_t len, int src) {
    sockaddr_storage addr;
    socklen_t addr_len;
    while (len != 0) {
      ssize_t r =
          recvfrom(socket_fd_, buf, len, 0, (struct sockaddr*)&addr, &addr_len);
      if (r <= 0) {
        throw std::runtime_error("Recv failed");
      }
      buf += r;
      len -= r;
    }
  }

 private:
  std::vector<address_t> peers_;
  int rank_;
  bool global_;
  int socket_fd_;
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
  if (group->size() != 2) {
    throw std::runtime_error("Only pairwise communication supported for now");
  }
  array input = ensure_row_contiguous(input_);
  if (input.data<void>() == output.data<void>()) {
    throw std::runtime_error("Donation not supported");
  } else {
    if (group->rank() == 0) {
      group->send(input.data<char>(), input.nbytes(), 1);
      group->recv(output.data<char>(), output.nbytes(), 1);
      sum_inplace(input, output);
    } else {
      group->recv(output.data<char>(), output.nbytes(), 0);
      group->send(input.data<char>(), input.nbytes(), 0);
      sum_inplace(input, output);
    }
  }
}

void all_gather(Group group_, const array& input_, array& output) {
  auto group = std::static_pointer_cast<GroupImpl>(group_.raw_group());
  if (group->size() != 2) {
    throw std::runtime_error("Only pairwise communication supported for now");
  }
  array input = ensure_row_contiguous(input_);
  if (group->rank() == 0) {
    group->send(input.data<char>(), input.nbytes(), 1);
    group->recv(output.data<char>() + input.nbytes(), input.nbytes(), 1);
    std::memcpy(output.data<char>(), input.data<char>(), input.nbytes());
  } else {
    group->recv(output.data<char>(), input.nbytes(), 0);
    group->send(input.data<char>(), input.nbytes(), 0);
    std::memcpy(
        output.data<char>() + input.nbytes(),
        input.data<char>(),
        input.nbytes());
  }
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
