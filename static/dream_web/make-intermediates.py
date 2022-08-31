import transformers

from ldm.dream.pngwriter import PngWriter
from ldm.simplet2i import T2I
from scripts.dream import SAMPLER_CHOICES

t2i = T2I()
transformers.logging.set_verbosity_error()
t2i.load_model()

prompt = "a warrior prince, highly detailed digital art"

step_writer = PngWriter('./outputs/intermediates/')
step_index = 1
prefix = step_writer.unique_prefix()

def image_progress(sample, step):
    global step_index
    seed = t2i.seed
    image = t2i.model._sample_to_image(sample)
    name = f'{prefix}.{seed}.{step_index}.png'
    metadata = f'{prompt} -S{seed} [intermediate]'
    step_writer.save_image_and_prompt_to_png(image, metadata, name)
    step_index += 1

t2i.prompt2png(
    prompt=prompt,
    outdir="outputs/results",
    steps=15
)
