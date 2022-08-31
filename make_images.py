import transformers
from ldm.simplet2i import T2I
from scripts.dream import SAMPLER_CHOICES

t2i = T2I()
transformers.logging.set_verbosity_error()
t2i.load_model()

for sampler in SAMPLER_CHOICES:
    print(f'{sampler=}')
    t2i.prompt2png(
        prompt=f"a warrior prince, highly detailed digital art by rossdraws",
        outdir="outputs/samples",
        sampler_name=sampler,
        seed=99999,
        steps=15,
        batch_size=2
    )
