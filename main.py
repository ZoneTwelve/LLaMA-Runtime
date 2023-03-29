#!/usr/bin/env python
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.ops import GlobalCounters
from tinygrad.jit import TinyJit
from llama import *
from extra.utils import fake_torch_load_zipped, get_child
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from os import getenv, listdir
import argparse

load_dotenv('./config/Share.env')
MachineConf = getenv('MACHINE')
#load_dotenv(getenv('MACHINE'))

WEIGHT_DIR = getenv('WEIGHT', '../../weights/LLaMA')
MODEL_SIZE = getenv('MODEL', '7B')
TOKENIZER_FILENAME = f'{WEIGHT_DIR}/tokenizer.model'

VOCAB_SIZE = 32000
MODEL_ARGS = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}

if __name__ == '__main__':
  Tensor.no_grad = True

  parser = argparse.ArgumentParser(description='LLaMA Runtime environment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument('--count', type=int, default=1000, help="MOBOBax number of tokens to generate")
  parser.add_argument('--personality', type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")

  parser.add_argument('--temperature', type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--profile', action='store_true', help="Output profile data to out.prof")
  parser.add_argument('--backend', type=str, default="METAL", help="Support [CPU, GPU, METAL, CLANG, LLVM, TORCH]")
  args = parser.parse_args()
  Device.DEFAULT = args.backend
  chatbot = args.prompt == None
  
  print(f"using {Device.DEFAULT} backend")
  from sentencepiece import SentencePieceProcessor
  sp_model = SentencePieceProcessor(model_file=str(TOKENIZER_FILENAME))
  assert sp_model.vocab_size() == VOCAB_SIZE


  model = Transformer(**MODEL_ARGS)
  WEIGHT_FILES_DIR = f'{WEIGHT_DIR}/{MODEL_SIZE}'
  WEIGHT_FILES = [ f'{WEIGHT_FILES_DIR}/{fn}' for fn in listdir(WEIGHT_FILES_DIR) if fn.startswith('consolidated') ]
  weights = []

  for WEIGHT_FILE in WEIGHT_FILES:
    with Timing("loaded weights in ", lambda et_ns: f", {GlobalCounters.mem_used/1e9:.2f} GB loaded at {GlobalCounters.mem_used/et_ns:.2f} GB/s"):
      weights.append(fake_torch_load_zipped(open(WEIGHT_FILE, "rb"), load_weights=getenv("WEIGHTS", 1)))
  set_of_weight = weights[0].keys()
  
  # Check the key of weights
  for w in weights:
    assert set_of_weight == w.keys()
  
  print("concat weights")

  for k,v in (t := tqdm(weights[0].items())):
    print( k, v )
    # assert GlobalCounters.mem_used/1e9 < 28, "used over 28 GB"
    t.set_description(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
    if 'rope.freqs' in k: continue  # no rope today
    mv = get_child(model, k)
    print(mv)
    #w0, w1 = v, weights1[k]

    # if the weight is copied across models, it's simple
    # TODO: assert they are the same
    #if w0.shape == mv.shape:
    #  mv.assign(w0)
    #  mv.realize()
    #  w1.lazydata.realized._buf = None
    #  continue

  #  if w0.shape[0] != mv.shape[0]: mv.assign(w0.cat(w1, dim=0))
  #  elif w0.shape[1] != mv.shape[1]: mv.assign(w0.cat(w1, dim=1))
  #  else: raise RuntimeError("what axis mismatch?")
  #  mv.realize()

    # rug the small tensor pieces
    #w0.lazydata.realized._buf = None
    #w1.lazydata.realized._buf = None
  
  for i in range(len(weight)):
    del weights[i]
