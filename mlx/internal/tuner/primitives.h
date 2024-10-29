// Copyright © 2024 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core::internal {

class TunableMatmul : public mlx::core::fast::Custom {
 public:
  TunableMatmul(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      std::unordered_map<std::string, int> tparams)
      : mlx::core::fast::Custom(stream, fallback), tparams_(tparams) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(TunableMatmul)

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
  std::unordered_map<std::string, int> tparams_;
};

} // namespace mlx::core::internal