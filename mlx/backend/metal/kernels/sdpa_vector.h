// Copyright © 2024 Apple Inc.

#include <metal_simdgroup>

using namespace metal;

template <typename T, int D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant float& scale,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 4;
  constexpr int elem_per_thread = D / BD;

  const int stride = BN * D;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + quad_lid * elem_per_thread;
  keys += kv_head_idx * k_stride + quad_gid * D + quad_lid * elem_per_thread;
  values += kv_head_idx * k_stride + quad_gid * D + quad_lid * elem_per_thread;
  out += head_idx * D + simd_gid * elem_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < elem_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (int i = quad_gid; i < N; i += BN) {
    // Read the key
    for (int i = 0; i < elem_per_thread; i++) {
      k[i] = keys[i];
    }

    // Compute the i-th score
    U score = 0;
    for (int i = 0; i < elem_per_thread; i++) {
      score += q[i] * k[i];
    }
    score = quad_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * values[i];
    }

    // Move the pointers to the next kv
    keys += stride;
    values += stride;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  // Each quadgroup communicates it's max score
  if (quad_lid == 0) {
    max_scores[quad_gid] = max_score;
    sum_exp_scores[quad_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    // 128 threads with 32 values per thread
    outputs[simd_gid * BN + simd_lid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_lid * BD + simd_gid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int group_size, int bits>
[[kernel]] void quant_sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device uint32_t* keys [[buffer(1)]],
    const device T* key_scales [[buffer(2)]],
    const device T* key_biases [[buffer(3)]],
    const device uint32_t* values [[buffer(4)]],
    const device T* value_scales [[buffer(5)]],
    const device T* value_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    const constant int& gqa_factor,
    const constant int& N,
    const constant size_t& k_stride,
    const constant size_t& group_stride,
    const constant float& scale,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint quad_gid [[quadgroup_index_in_threadgroup]],
    uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 4;
  constexpr int elem_per_thread = D / BD;
  constexpr int pack_factor = 32 / bits;

  const int stride = BN * D;

  typedef float U;

  thread U q[elem_per_thread];
  thread U k[elem_per_thread];
  thread U v[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  queries += head_idx * D + quad_lid * elem_per_thread;

  const int kv_idx = quad_gid * D + quad_lid * elem_per_thread;
  const int packed_idx = kv_head_idx * k_stride + kv_idx / pack_factor;
  const int group_idx = kv_head_idx * group_stride + kv_idx / group_size;
  keys += packed_idx;
  key_scales += group_idx;
  key_biases += group_idx;
  values += packed_idx;
  value_scales += group_idx;
  value_biases += group_idx;

  out += head_idx * D + simd_gid * elem_per_thread;

  // Read the query and 0 the output accumulator
  U query_sum = 0;
  U shifts[4] = {1, 16, 256, 4096};
  for (int i = 0; i < elem_per_thread; i++) {
    // Shift by the appropriate amount here
    query_sum += queries[i];
    U shift = shifts[i % 4];
    q[i] = static_cast<U>(scale) * queries[i] / shift;
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (int i = quad_gid; i < N; i += BN) {
    // Read the key
    auto ks = (const device uint16_t*)keys;
    for (int i = 0; i < elem_per_thread / 4; i++) {
      k[4 * i] = ks[i] & 0x000f;
      k[4 * i + 1] = ks[i] & 0x00f0;
      k[4 * i + 2] = ks[i] & 0x0f00;
      k[4 * i + 3] = ks[i] & 0xf000;
    }
    // All the keys in a set are in the same group
    U key_scale = key_scales[0];
    U key_bias = key_biases[0];

    // Compute the i-th score
    U score = 0;
    for (int i = 0; i < elem_per_thread; i++) {
      score += q[i] * k[i];
    }
    score = score * key_scale + query_sum * key_bias;
    score = quad_sum(score);

    // Update the accumulators
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    U value_scale = value_scales[0];
    U value_bias = value_biases[0];

    // Load the values
    auto vs = (const device uint16_t*)values;
    U s[4] = {
        value_scale,
        value_scale / 16.0f,
        value_scale / 256.0f,
        value_scale / 4096.0f};
    for (int i = 0; i < elem_per_thread / 4; i++) {
      v[4 * i] = s[0] * (vs[i] & 0x000f) + value_bias;
      v[4 * i + 1] = s[1] * (vs[i] & 0x00f0) + value_bias;
      v[4 * i + 2] = s[2] * (vs[i] & 0x0f00) + value_bias;
      v[4 * i + 3] = s[3] * (vs[i] & 0xf000) + value_bias;
    }

    // Update the output accumulator
    for (int i = 0; i < elem_per_thread; i++) {
      o[i] = o[i] * factor + exp_score * v[i];
    }

    // Move the pointers to the next kv
    keys += stride / pack_factor;
    key_scales += stride / group_size;
    key_biases += stride / group_size;
    values += stride / pack_factor;
    value_scales += stride / group_size;
    value_biases += stride / group_size;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  // Each quadgroup communicates it's max score
  if (quad_lid == 0) {
    max_scores[quad_gid] = max_score;
    sum_exp_scores[quad_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < elem_per_thread; i++) {
    // 128 threads with 32 values per thread
    outputs[simd_gid * BN + simd_lid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_lid * BD + simd_gid] * factor) / sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
