import argparse
import math
from functools import partial

import mlx.core as mx
from time_utils import time_fn

D = 16384
group_size = 64
bits = 3
dtype = mx.float16
loops = 10


def qmv_(x, wq, q_type):
    for i in range(loops):
        x = mx.quantized_matmul(
            x,
            *wq,
            group_size=group_size,
            bits=bits,
            type=q_type,
        )
    return x


def affine_qmv(x, wq):
    return qmv_(x, wq, "affine")


def affine_packed_qmv(x, wq):
    return qmv_(x, wq, "affine-packed")


def time_qmv():
    mx.random.seed(3)
    x = mx.random.normal(shape=(1, D)).astype(dtype)
    w = mx.random.normal(shape=(D, D)).astype(dtype)
    wq = mx.quantize(w, group_size=group_size, bits=bits, type="affine")
    mx.eval(x, wq)
    time_fn(affine_qmv, x, wq)


def time_packed_qmv():
    mx.random.seed(3)
    x = mx.random.normal(shape=(1, D)).astype(dtype)
    w = mx.random.normal(shape=(D, D)).astype(dtype)
    wq = mx.quantize(w, group_size=group_size, bits=bits, type="affine-packed")
    mx.eval(x, wq)
    time_fn(affine_packed_qmv, x, wq)


if __name__ == "__main__":
    time_qmv()
    time_packed_qmv()
