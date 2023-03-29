#!/usr/bin/env python
# coding: utf-8

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
import sys, argparse, math

load_dotenv('./config/Share.env')
#MachineConf = getenv('MACHINE')
load_dotenv(getenv('MACHINE'))

WEIGHT_DIR = getenv('WEIGHT', '../../weights/LLaMA')
MODEL_SIZE = getenv('MODEL', '7B')
TOKENIZER_FILENAME = f'{WEIGHT_DIR}/tokenizer.model'
MODEL_DIM = {
    '7B': 4096,
    '13B': 5120,
    '33B': 6656,
    '65B': 8192,
}[MODEL_SIZE]

VOCAB_SIZE = 32000
MODEL_ARGS = {"dim": MODEL_DIM, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}

print(f"Run with {MODEL_SIZE} model")

from llama import *


if __name__ == '__main__':
  Tensor.no_grad = True

  parser = argparse.ArgumentParser(description='LLaMA Runtime environment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument('--count', type=int, default=1000, help="MOBOBax number of tokens to generate")

  parser.add_argument('--temperature', type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--profile', action='store_true', help="Output profile data to out.prof")
  parser.add_argument('--backend', type=str, default="METAL", help="Support [CPU, GPU, METAL, CLANG, LLVM, TORCH]")
  #args = parser.parse_args()
  args = parser.parse_args(args=[]) # for jupyter
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
  t.set_description(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
  if 'rope.freqs' in k: continue
  
  mv = get_child(model, k)
  w = [v] + [ _w[k] for _w in weights[1:] ]
  
  if w[0].shape == mv.shape:
    mv.assign(w[0])
    mv.realize()
    for _w in w[1:]:
      _w.lazydata.realized._buf = None
    continue
  for _w in w[1:]:
    dim = -1
    if w[0].shape[0] != mv.shape[0]:
      dim = 0
    elif w[0].shape[1] != mv.shape[1]:
      dim = 1
    if dim == -1:
      raise RuntimeError("what axis mismatch?")
    mv.assign(w[0].cat(_w, dim=dim))
  mv.realize()

# Delete the weights
for i in range(len(w)):
    del w[i]
print("concatenated")
    
pre_prompt = """Consider that the following is conversation between AI assistant named Gale and human named USER.
You are Wilson.
You love to answer questions and you are very good at it.
You will write on language that USER asks you to write on.
You are fluent in any language.

YOUR INSTRUCTIONS:
-------------------------------------------------------------------------------
You may only interact with the [USER] in form of the following commands:
"RESP" "message" [EOS]

CHAT HISTORY:
-------------------------------------------------------------------------------
USER: what is the population of Taiwan?
RESP: The current population of Taiwan is 23,934,280 as of Tuesday, March 28, 2023, based on Worldometer elaboration of the latest United Nations data.[EOS]
USER: What's the highest building in Taiwan?
RESP: The tallest building in Taiwan is currently the 101–story Taipei 101, which rises 509.2 metres (1,671 ft) and was completed in 2004. It was officially classified as the world's tallest from 2004 to 2010. Now, it is still the tallest building in Taiwan.[EOS]
USER: How do I make an HTTP request in Javascript?
RESP: In JavaScript, you can make an HTTP request using the built-in fetch() function. Here's an example of how to use fetch() to make a GET request to a URL and retrieve the response:
```javascript
fetch('https://example.com/data.json')
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error(error));

```
In this example, fetch() takes a URL as its first parameter and returns a promise that resolves to a Response object. You can then use the json() method of the Response object to parse the response body as JSON. The resulting data is then logged to the console. If there is an error, it is caught and logged to the console.[EOS]
USER: What is the color of sky?
RESP: Color of sky can be various shades of blue.[EOS]
"""

USER_DELIM = getenv('USER_DELIM', 'USER:')
RESP_DELIM = getenv('RESP_DELIM', 'RESP:')
END_DELIM = getenv('END_DELIM', '[EOS]')

def onehot_encode(toks, vocab_size=VOCAB_SIZE):
  # this allows the embedding to work in tinygrad
  onehot = np.zeros((1, len(toks), vocab_size), dtype=np.float32)
  onehot[0,range(len(toks)),toks] = 1
  return Tensor(onehot)

def sample(logits, temperature):
  if temperature < 1e-6:
    # so close to 0 we use argmax
    return int(logits.numpy().argmax())
  else:
    probs = (logits / temperature).softmax()
    probs = probs.numpy().flatten()
    return int(np.random.choice(len(probs), p=probs))

if chatbot:
  # encode pre prompt
  toks = [sp_model.bos_id()] + sp_model.encode(pre_prompt)

  print(f"Run thought KV cache for the transformers...")
  with Timing():
    model(onehot_encode(toks), 0).realize()  # NOTE: output logits are not used
    start_pos = len(toks)
else:
  # non chat bot mode
  toks = [sp_model.bos_id()] + sp_model.encode(args.prompt)
  start_pos = 0

# print prompt
outputted = sp_model.decode(toks)
sys.stdout.write(outputted)
sys.stdout.flush()


while True:
  if chatbot:
    user_prompt = USER_DELIM + input(USER_DELIM) + "\n"
    outputted += USER_DELIM

  new_toks = [sp_model.bos_id()] + sp_model.encode(outputted)
  assert toks == new_toks[:len(toks)]
  toks = new_toks
  assert outputted == sp_model.decode(toks)

  last_break = len(outputted)
  for i in range(args.count):
    if args.profile and i == 2: profiler.enable()

    if args.timing: print("")
    st = GlobalCounters.time_sum_s
    with Timing("ran model in ", on_exit=(lambda et: f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU") if DEBUG else None, enabled=args.timing):
      logits = model(onehot_encode(toks[start_pos:]), start_pos).realize()
    with Timing("sync in ", enabled=args.timing):
      tok = sample(logits, args.temperature)

    # use the kv cache
    start_pos = len(toks)

    # add the new token
    toks.append(tok)

    # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
    cur = sp_model.decode(toks)
    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur
    if chatbot and outputted.endswith(END_DELIM): break

  if not chatbot: break

