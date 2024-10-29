// Copyright © 2024 Apple Inc.

#include "mlx/internal/tuner/ops.h"
#include "mlx/internal/tuner/primitives.h"
#include "mlx/ops.h"

namespace mlx::core::internal {

array tunable_matmul(
    const array& in_a,
    const array& in_b,
    const std::unordered_map<std::string, int>& tparams,
    StreamOrDevice s_ /*= {} */) {
  auto s = to_stream(s_);
  auto fallback = [s](const std::vector<array>& inputs) {
    return std::vector<array>{matmul(inputs[0], inputs[1], s)};
  };

  if (s.device == Device::cpu || in_a.ndim() < 2 || in_b.ndim() < 2) {
    return matmul(in_a, in_b, s);
  }

  auto a = in_a;
  auto b = in_b;

  if (a.shape(-1) != b.shape(-2)) {
    std::ostringstream msg;
    msg << "[matmul] Last dimension of first input with shape " << a.shape()
        << " must match second to last dimension of"
        << " second input with shape " << b.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto out_type = promote_types(a.dtype(), b.dtype());
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[matmul] Only real floating point types are supported but "
        << a.dtype() << " and " << b.dtype() << " were provided which results"
        << " in " << out_type << ", which is not a real floating point type.";
    throw std::invalid_argument(msg.str());
  }

  a = astype(a, out_type, s);
  b = astype(b, out_type, s);

  // We can batch the multiplication by reshaping a
  if (a.ndim() > 2 && b.ndim() == 2) {
    std::vector<int> out_shape = a.shape();
    a = reshape(a, {-1, out_shape.back()}, s);
    out_shape.back() = b.shape(-1);
    if (in_b.ndim() == 1) {
      out_shape.pop_back();
    }
    auto out = array(
        {a.shape(0), b.shape(1)},
        out_type,
        std::make_shared<TunableMatmul>(to_stream(s), fallback, tparams),
        {a, b});
    return reshape(out, out_shape, s);
  }

  if (a.ndim() > 2 || b.ndim() > 2) {
    std::vector<int> bsx_a(a.shape().begin(), a.shape().end() - 2);
    std::vector<int> bsx_b(b.shape().begin(), b.shape().end() - 2);
    auto inner_shape = broadcast_shapes(bsx_a, bsx_b);

    // Broadcast a
    inner_shape.push_back(a.shape(-2));
    inner_shape.push_back(a.shape(-1));
    a = broadcast_to(a, inner_shape, s);

    // Broadcast b
    *(inner_shape.end() - 2) = b.shape(-2);
    *(inner_shape.end() - 1) = b.shape(-1);
    b = broadcast_to(b, inner_shape, s);
  }

  auto out_shape = a.shape();
  out_shape.back() = b.shape(-1);

  return array(
      std::move(out_shape),
      out_type,
      std::make_shared<TunableMatmul>(to_stream(s), fallback, tparams),
      {a, b});
}

} // namespace mlx::core::internal