// Copyright © 2024 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/utils.h"

namespace mlx::core::internal {

array tunable_matmul(
    const array& a,
    const array& b,
    const std::unordered_map<std::string, int>& tparams,
    StreamOrDevice s = {});

} // namespace mlx::core::internal