// Copyright © 2024 Apple Inc.

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

#include "gloo/allreduce.h"
#include "gloo/math.h"
#include "gloo/mpi/context.h"
#include "gloo/transport/uv/device.h"

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

namespace mlx::core::distributed {

namespace {
array ensure_row_contiguous(const array& arr) {
  if (arr.flags().row_contiguous) {
    return arr;
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy(arr, arr_copy, CopyType::General);
    return arr_copy;
  }
}
} // namespace

bool is_available() {
  return true;
}

int Group::rank() {
  return std::static_pointer_cast<gloo::mpi::Context>(group_)->rank;
}

int Group::size() {
  return std::static_pointer_cast<gloo::mpi::Context>(group_)->size;
}

Group Group::split(int color, int key) {
  throw std::runtime_error("split is NYI");
}

void Group::barrier() {
  throw std::runtime_error("barrier is NYI");
}

struct GlooCTX {
  std::shared_ptr<gloo::mpi::Context> context;
  std::shared_ptr<gloo::transport::Device> dev;
};

Group init(bool strict /* = false */) {
  static std::shared_ptr<GlooCTX> gloo_ctx = nullptr;

  if (gloo_ctx == nullptr) {
    gloo_ctx = std::make_shared<GlooCTX>();
    gloo_ctx->context = gloo::mpi::Context::createManaged();
    gloo_ctx->dev = gloo::transport::uv::CreateDevice("localhost");
    gloo_ctx->context->connectFullMesh(gloo_ctx->dev);
  }
  return Group(gloo_ctx->context);
}

namespace detail {

Stream communication_stream() {
  static Stream comm_stream = new_stream(Device::cpu);
  return comm_stream;
}

template <typename T>
void all_reduce_sum(
    std::shared_ptr<gloo::mpi::Context> context,
    T* output,
    T* input,
    size_t len) {
  gloo::AllreduceOptions opts_(context);
  opts_.setInput(input, len);
  opts_.setOutput(output, len);
  opts_.setAlgorithm(gloo::AllreduceOptions::Algorithm::RING);
  opts_.setReduceFunction(
      static_cast<void (*)(void*, const void*, const void*, size_t)>(
          &gloo::sum<T>));
  gloo::allreduce(opts_);
}

void all_sum(Group group_, const array& input_, array& output) {
  array input = ensure_row_contiguous(input_);
  if (input.data<void>() != output.data<void>()) {
    std::memcpy(output.data<char>(), input.data<char>(), input.nbytes());
  }
  auto context =
      std::static_pointer_cast<gloo::mpi::Context>(group_.raw_group());
  SWITCH_TYPE(
      output,
      all_reduce_sum<T>(
          context, output.data<T>(), input.data<T>(), input.size()));
}

void all_gather(Group group_, const array& input_, array& output) {
  throw std::runtime_error("all_gather NYI");
}

void send(Group group_, const array& input_, int dst) {
  throw std::runtime_error("send NYI");
}

void recv(Group group_, array& out, int src) {
  throw std::runtime_error("recv NYI");
}

} // namespace detail

} // namespace mlx::core::distributed
