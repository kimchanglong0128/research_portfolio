import torch

# 自动选择设备
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

from diffusion_demo.diffusion.scheduler import DiffusionScheduler
from diffusion_demo.models.unet_cond import UNetConditional
from diffusion_demo.models.clip_conditioner import CLIPTextConditioner


def main():
    # 1. scheduler
    scheduler = DiffusionScheduler(timesteps=1000, device=device)

    # 2. CLIP conditional model
    clip_cond = CLIPTextConditioner(device=device)

    # 3. UNet conditional model
    unet = UNetConditional(cond_dim=clip_cond.cond_dim).to(device)

    # 4. fake data for testing
    B = 4
    x0 = torch.randn(B, 3, 32, 32, device=device)
    t = torch.randint(0, scheduler.timesteps, (B,), device=device)

    x_t, noise = scheduler.q_sample(x0, t)

    prompts = ["a small airplane", "a cute cat", "a red car", "a dog on the grass"]
    cond_tokens = clip_cond(prompts)  # [B, 1, cond_dim]

    pred_noise = unet(x_t, t, cond_tokens)

    print("x_t shape:       ", x_t.shape)
    print("cond_tokens shape", cond_tokens.shape)
    print("pred_noise shape:", pred_noise.shape)


if __name__ == "__main__":
    main()
