from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DreamParameters():
    # these uniquely determine an image for txt2img, within the constraints of other values which are constant in our system
    # other constants are ddim_eta (0), downsampling_factor (8), and latent_channels (4)
    # for img2img, the initial image and a strength are also necessary
    # also, this assumes --batch_size=1; supporting other batch sizes is tricky
    sampler_name: str
    width: int
    height: int
    cfg_scale: float
    steps: int
    seed: int
    weighted_prompts: List[Tuple[str, float]] # weighted list of prompts
