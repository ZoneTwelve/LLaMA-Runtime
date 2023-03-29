import numpy as np
np.set_printoptions(linewidth=200)
from typing import Optional, Tuple

from tinygrad.helpers import getenv, DEBUG
from tinygrad.lazy import Device

from extra.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.ops import GlobalCounters
from tinygrad.jit import TinyJit

import math

# https://github.com/facebookresearch/llama/blob/57b0eb62de0636e75af471e49e2f1862d908d9d8/llama/model.py#L47-L52
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[:(dim // 2)] / dim))
  freqs = np.outer(np.arange(end, dtype=np.float32), freqs)
  return np.stack([np.cos(freqs), np.sin(freqs)], axis=-1).reshape(1, end, 1, dim//2, 2)

# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, B):
  assert len(A.shape) == 5 and len(B.shape) == 5
  a,b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
  c,d = B[:, :, :, :, 0:1], B[:, :, :, :, 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  xq_out = complex_mult(xq, freqs_cis)
  xk_out = complex_mult(xk, freqs_cis)
  return xq_out.flatten(3), xk_out.flatten(3)

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight

class Attention:
  def __init__(self, dim, n_heads):
    self.wq, self.wk, self.wv, self.wo = [Linear(dim, dim, bias=False) for _ in range(4)]
    self.n_heads = n_heads
    self.head_dim = dim // n_heads

  def prepare_attention(self, x:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim) for x in (xq, xk, xv)]
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    return xq, xk, xv

  def inner_attention(self, xq:Tensor, xk:Tensor, xv:Tensor, start_pos:int, mask:Optional[Tensor]) -> Tensor:
    bsz, seqlen, _, _ = xq.shape
    # kv caching!
    if start_pos == 0:
      keys, values = xk, xv
    else:
      assert hasattr(self, 'cache_k'), "no cache"
      assert start_pos == self.cache_k.shape[1] and start_pos == self.cache_v.shape[1], "cache is wrong shape"
      assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
      keys, values = self.cache_k.cat(xk, dim=1), self.cache_v.cat(xv, dim=1)

    # save the cache
    self.cache_k, self.cache_v = keys.realize(), values.realize()

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = xq.matmul(keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask
    scores = scores.softmax()  # this is casted to float
    return scores.matmul(values).transpose(1, 2).reshape(bsz, seqlen, -1)

  # NOTE: this is not called
  def __call__(self, x:Tensor, start_pos:int, freqs_cis:Tensor, mask:Optional[Tensor]) -> Tensor:
    xq, xk, xv = self.prepare_attention(x, freqs_cis)
    output = self.inner_attention(xq, xk, xv, start_pos, mask)
    return self.wo(output)

class FeedForward:
  def __init__(self, dim, hidden_dim, multiple_of):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x))

class TransformerBlock:
  def __init__(self, dim, multiple_of, n_heads, norm_eps):
    self.attention = Attention(dim, n_heads)
    self.feed_forward = FeedForward(dim, 4*dim, multiple_of)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)
    if getenv("JIT"):
      self._pre = TinyJit(self.pre)
      self._post = TinyJit(self.post)
    else:
      self._pre, self._post = self.pre, self.post

  def pre(self, x:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xq, xk, xv = self.attention.prepare_attention(self.attention_norm(x), freqs_cis)
    return xq.realize(), xk.realize(), xv.realize()

  def post(self, x:Tensor, output:Tensor) -> Tensor:
    h = x + self.attention.wo(output)
    return (h + self.feed_forward(self.ffn_norm(h))).realize()

  def __call__(self, x:Tensor, start_pos:int, freqs_cis:Tensor, mask:Optional[Tensor]):
    xq, xk, xv = self._pre(x, freqs_cis)
    # inner_attention can't be jitted because it's dynamic based on start_pos
    output = self.attention.inner_attention(xq, xk, xv, start_pos, mask)
    return self._post(x, output)

class Transformer:
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, max_batch_size=32, max_seq_len=1024):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, norm_eps) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    self.tok_embeddings = {"weight": Tensor.glorot_uniform(vocab_size, dim)}
    self.output = Linear(dim, vocab_size, bias=False)
    self.freqs_cis = Tensor(precompute_freqs_cis(dim // n_heads, max_seq_len * 2))

  def __call__(self, tokens:Tensor, start_pos:int):
    _bsz, seqlen, _ = tokens.shape
    h = tokens @ self.tok_embeddings['weight']

    # get only the part we are using. making it contiguous avoids more kernel calls
    freqs_cis = self.freqs_cis[:, start_pos:start_pos+seqlen].contiguous().realize()

    if seqlen > 1:
      mask = np.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=np.float32)
      mask = np.triu(mask, k=start_pos + 1)  # TODO: this is hard to do in tinygrad
      mask = Tensor(mask)
    else:
      mask = None

    for layer in self.layers:
      h.realize()  # TODO: why do i need this?
      h = layer(h, start_pos, freqs_cis, mask)

    return self.output(self.norm(h)[:, -1, :])

