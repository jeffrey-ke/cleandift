import diffusers
import torch
from torch import nn

class AutoencoderKL(nn.Module):
    def __init__(
        self,
        scale: float = 0.18215,
        shift: float = 0.0,
        repo: str = "sd2-community/stable-diffusion-2-1",
        dtype: str = "bfloat16",
    ):
        super().__init__()
        self.scale = scale
        self.shift = shift
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.ae = diffusers.AutoencoderKL.from_pretrained(
            repo, subfolder="vae", torch_dtype=torch_dtype
        )
        self.ae.to(dtype=torch_dtype)
        self.ae.eval()
        self.ae.compile()
        self.ae.requires_grad_(False)

    def forward(self, img):
        return self.encode(img)

    @torch.no_grad()
    def encode(self, img):
        latent = self.ae.encode(img, return_dict=False)[0].sample()
        return (latent - self.shift) * self.scale

    @torch.no_grad()
    def decode(self, latent):
        rec = self.ae.decode(latent / self.scale + self.shift, return_dict=False)[0]
        return rec
