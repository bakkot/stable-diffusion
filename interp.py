from ldm.simplet2i import T2I
import transformers
import numpy as np
import torch
from pytorch_lightning import seed_everything
import random

t2i = T2I()

transformers.logging.set_verbosity_error()

print("loading model...")
t2i.load_model()

prompt = "elf queen with rainbow hair, golden hour. colored pencil drawing by rossdraws andrei riabovitchev trending on artstation"

print("interpolating...")
t2i.dumb_interp(img_1 = "inputs/elf.png", img_2 = "inputs/elf2.png")
