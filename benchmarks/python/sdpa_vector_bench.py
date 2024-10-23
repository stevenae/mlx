import mlx.core as mx
import numpy as np
from time_utils import time_fn

L = 16
H = 32
H_k = 32 // 4
D = 128


def attention(q, k, v):
    k = mx.quantize(k)
    v = mx.quantize(v)
    k = mx.dequantize(*k)
    v = mx.dequantize(*v)
    B, Hq, L, D = q.shape
    _, Hk, S, _ = k.shape
    q = q.reshape(B, Hk, Hq // Hk, L, D)
    k = k[:, :, None, :, :]
    v = v[:, :, None, :, :]
    s = q @ k.transpose(0, 1, 2, 4, 3)
    p = mx.softmax(s.astype(mx.float32), axis=-1).astype(s.dtype)
    o = p @ v
    return o.reshape(B, Hq, L, D)


def sdpa(q, k, v):
    k = mx.quantize(k, bits=8)
    v = mx.quantize(v, bits=8)
    k = mx.dequantize(*k, bits=8)
    v = mx.dequantize(*v, bits=8)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=None)


def quant_sdpa(q, k, v):
    k = mx.quantize(k, bits=8)
    v = mx.quantize(v, bits=8)
    return mx.fast.quantized_scaled_dot_product_attention(
        q, *k, *v, scale=1.0, mask=None, bits=8
    )


def time_self_attention_primitives(q, k, v):
    time_fn(attention, q, k, v)


def time_self_attention_sdpa(q, k, v):
    time_fn(sdpa, q, k, v)


def time_self_attention_quant_sdpa(q, k, v):
    time_fn(quant_sdpa, q, k, v)


if __name__ == "__main__":
    mx.random.seed(3)
    # q = mx.random.uniform(shape=(1, H, 1, D))
    # k = mx.random.uniform(shape=(1, H_k, L, D))
    # v = mx.random.uniform(shape=(1, H_k, L, D))
    q = mx.array(np.load("/Users/alexbarron/mlx-examples/llms/queries.npy"))
    k = mx.array(np.load("/Users/alexbarron/mlx-examples/llms/keys.npy"))
    v = mx.array(np.load("/Users/alexbarron/mlx-examples/llms/values.npy"))
    print(q.dtype)
    print(q.shape, k.shape, v.shape)
    mx.eval(q, k, v)

    k_quant = mx.quantize(k)
    v_quant = mx.quantize(v)
    mx.eval(k_quant, v_quant)

    # time_self_attention_sdpa(q, k, v)
    # time_self_attention_quant_sdpa(q, k_quant, v_quant)
    # time_self_attention_primitives(q, k, v)
    q_sdpa = quant_sdpa(q, k, v)
    print(q_sdpa)
    # o_attention = attention(q, k, v)
    # print(o_attention)
    # np.testing.assert_allclose(q_sdpa, o_attention, atol=1e-5)
    o_sdpa = sdpa(q, k, v)
    print(o_sdpa)
    np.testing.assert_allclose(q_sdpa, o_sdpa, atol=1e-5)
    # print(o_sdpa[..., :64])
    # print()
    # print(o_attention[..., :64])
    # np.testing.assert_allclose(o_sdpa, o_attention)
