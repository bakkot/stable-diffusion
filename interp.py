from ldm.simplet2i import T2I
import transformers
import numpy as np
import torch
from pytorch_lightning import seed_everything
import random

t2i = T2I(
    latent_diffusion_weights=False,
    config  = "configs/stable-diffusion/v1-inference.yaml"
)


transformers.logging.set_verbosity_error()

print("loading model...")
t2i.load_model()

prompt = "elf queen with rainbow hair, golden hour. colored pencil drawing by rossdraws andrei riabovitchev trending on artstation"

print("interpolating...")
t2i.imgs2img(prompt, init_img_1 = "inputs/elf.png", init_img_2 = "inputs/elf2.png", iterations = 40)
