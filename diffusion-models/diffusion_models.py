"""Diffusion Models: code from the ML Zero to Hero explainer.

Video: https://www.youtube.com/@mlzerotoherousa
"""

# Minimal DDPM from scratch on a 2-D toy distribution (runs on CPU in ~1 minute).
import math, torch, torch.nn as nn
torch.manual_seed(0)

def sample_data(n):                       # 8 Gaussians on a ring
    k = torch.randint(0, 8, (n,))
    ang = k.float() * (2 * math.pi / 8)
    centers = torch.stack([torch.cos(ang), torch.sin(ang)], 1) * 2.0
    return centers + 0.1 * torch.randn(n, 2)

T = 200
betas = torch.linspace(1e-4, 0.05, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)

class Eps(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 + 16, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t):              # t: integer timesteps, shape (B,)
        freqs = torch.exp(-math.log(1000) * torch.arange(8) / 8)
        ang = t[:, None].float() * freqs[None]
        temb = torch.cat([torch.sin(ang), torch.cos(ang)], 1)
        return self.net(torch.cat([x, temb], 1))

model = Eps()
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
for step in range(4000):
    x0 = sample_data(512)
    t = torch.randint(0, T, (512,))
    eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
    loss = ((model(xt, t) - eps) ** 2).mean()          # L_simple
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"step {step}  loss {loss.item():.4f}")

@torch.no_grad()
def sample(n):
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps_hat = model(x, tt)
        mean = (x - betas[t] / (1 - abar[t]).sqrt() * eps_hat) / alphas[t].sqrt()
        x = mean + (betas[t].sqrt() * torch.randn_like(x) if t > 0 else 0)
    return x

xs = sample(2000)
radius = xs.norm(dim=1)
print("mean radius (target 2.0):", radius.mean().item().__round__(3))

# --------------------------------------------------------------------------

# The same training step with Hugging Face diffusers building blocks (images, U-Net backbone).
import torch
from diffusers import DDPMScheduler, UNet2DModel
torch.manual_seed(0)

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3,
                   block_out_channels=(64, 128, 128), layers_per_block=1,
                   down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                   up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"))
opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)

clean = torch.rand(8, 3, 32, 32) * 2 - 1                 # stand-in for a batch of images in [-1, 1]
noise = torch.randn_like(clean)
t = torch.randint(0, scheduler.config.num_train_timesteps, (clean.shape[0],))
noisy = scheduler.add_noise(clean, noise, t)            # sqrt(abar) x0 + sqrt(1-abar) eps
loss = torch.nn.functional.mse_loss(unet(noisy, t).sample, noise)
loss.backward(); opt.step()
print("loss", loss.item())
# Sampling: from diffusers import DDPMPipeline; DDPMPipeline(unet=unet, scheduler=scheduler)(num_inference_steps=1000)
