import argparse
import math
from functools import partial

import mlx.core as mx
from time_utils import time_fn

D = 1024
M = 4 * D
group_size = 64
bits = 4
dtype = mx.float16
loops = 100


def qmv_(x, wq1, wq2, q_type):
    for i in range(loops):
        x = mx.quantized_matmul(
            x,
            *wq1,
            group_size=group_size,
            bits=bits,
            type=q_type,
        )
        x = mx.quantized_matmul(
            x,
            *wq2,
            group_size=group_size,
            bits=bits,
            type=q_type,
        )
    return x


def affine_qmv(x, wq1, wq2):
    return qmv_(x, wq1, wq2, "affine")


def affine_packed_qmv(x, wq1, wq2):
    return qmv_(x, wq1, wq2, "affine-packed")


def time_qmv():
    mx.random.seed(3)
    x = mx.random.normal(shape=(1, D)).astype(dtype)
    w1 = mx.random.normal(shape=(M, D)).astype(dtype)
    wq1 = mx.quantize(w1, group_size=group_size, bits=bits, type="affine")
    w2 = mx.random.normal(shape=(D, M)).astype(dtype)
    wq2 = mx.quantize(w2, group_size=group_size, bits=bits, type="affine")
    mx.eval(x, wq1, wq2)
    time_fn(affine_qmv, x, wq1, wq2)


def time_packed_qmv():
    mx.random.seed(3)
    x = mx.random.normal(shape=(1, D)).astype(dtype)
    w1 = mx.random.normal(shape=(M, D)).astype(dtype)
    wq1 = mx.quantize(w1, group_size=group_size, bits=bits, type="affine-packed")
    w2 = mx.random.normal(shape=(D, M)).astype(dtype)
    wq2 = mx.quantize(w2, group_size=group_size, bits=bits, type="affine-packed")
    mx.eval(x, wq1, wq2)
    time_fn(affine_packed_qmv, x, wq1, wq2)


if __name__ == "__main__":
    for b in [2, 3, 4, 6, 8]:
        bits = b
        print(f"Bits {bits}:")
        time_qmv()
        time_packed_qmv()
