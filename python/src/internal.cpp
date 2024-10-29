// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "python/src/utils.h"

#include "mlx/internal/tuner/ops.h"
#include "mlx/ops.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

void init_internal(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "internal", "mlx.core.internal: internal operations");

  m.def(
      "tunable_matmul",
      &internal::tunable_matmul,
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tunable_matmul(a: array, b: array, tparams: dict[str, int], /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Matrix multiplication.

        Perform the (possibly batched) matrix multiplication of two arrays. This function supports
        broadcasting for arrays with more than two dimensions.

        - If the first array is 1-D then a 1 is prepended to its shape to make it
          a matrix. Similarly if the second array is 1-D then a 1 is appended to its
          shape to make it a matrix. In either case the singleton dimension is removed
          from the result.
        - A batched matrix multiplication is performed if the arrays have more than
          2 dimensions.  The matrix dimensions for the matrix product are the last
          two dimensions of each input.
        - All but the last two dimensions of each input are broadcast with one another using
          standard numpy-style broadcasting semantics.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            tparams (dict[str, int]): Matmul tunable parameters

        Returns:
            array: The matrix product of ``a`` and ``b``.
      )pbdoc");
}