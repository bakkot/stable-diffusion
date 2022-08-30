import transformers
from ldm.simplet2i import T2I
from scripts.dream import SAMPLER_CHOICES

t2i = T2I()
transformers.logging.set_verbosity_error()
t2i.load_model()

for sampler in SAMPLER_CHOICES:
    print(sampler)
    t2i.prompt2png(f"a warrior prince, highly detailed digital art by rossdraws --sampler={sampler} --seed=999 --steps=20 --batch_size=2", "outputs/samples")
