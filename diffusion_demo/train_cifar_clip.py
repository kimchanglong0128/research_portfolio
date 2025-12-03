import os 
import toml 
import torch 
import torch.nn as nn
import torch.optim as optim

import wandb 
from torchvision.utils import make_grid

from diffusion_demo.diffusion.scheduler import DiffusionScheduler
from diffusion_demo.models.unet_cond import UNetConditional
from diffusion_demo.models.clip_conditioner import CLIPTextConditioner
from diffusion_demo.utils.data import get_dataloader, CIFAR10_CLASSES_PROMPTS

# Utility function to select device based on config and availability
def select_device(config):
    if config['training'].get('prefer_mps', True) and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

# Sampling function (for logging)
@torch.no_grad()
def sample_images(unet, scheduler, clip_cond, device, num_samples=8, steps=100):
    unet.eval()

    # all_labels = list(CIFAR10_CLASSES_PROMPTS.keys())
    # prompts = [CIFAR10_CLASSES_PROMPTS[all_labels[idx % len(all_labels)]] for idx in range(num_samples)]

    num_classes = len(CIFAR10_CLASSES_PROMPTS)                           
    prompts = [
        CIFAR10_CLASSES_PROMPTS[idx % num_classes]                       
        for idx in range(num_samples)
    ]    

    cond_tokens = clip_cond(prompts).to(device)  # [B, L, cond_dim]

    x = torch.randn(num_samples, 3, 32, 32).to(device)  # start from noise
    T = scheduler.timesteps
    steps = min(steps, T)
    timesteps = torch.linspace(T-1, 0, steps, dtype=torch.long).to(device)

    for i, t_scalar in enumerate(timesteps):
        t = t_scalar.long()
        t_batch = torch.full((num_samples,), t, dtype=torch.long).to(device)

        eps = unet(x, t_batch, cond_tokens)  # predict noise

        a_t = scheduler.alphas[t_batch].view(num_samples, 1, 1, 1)
        a_bar_t = scheduler.alpha_cumprod[t_batch].view(num_samples, 1, 1, 1)
        beta_t = scheduler.betas[t_batch].view(num_samples, 1, 1, 1)

        x0_pred = (x - torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(a_bar_t + 1e-8)

        if i == steps - 1:
            x = x0_pred
            break

        noise = torch.randn_like(x)
        a_bar_prev = scheduler.alpha_cumprod[torch.clamp(t_batch - 1, min=0)].view(num_samples, 1, 1, 1)

        x = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1 - a_bar_prev) * noise

    x = x.clamp(-1, 1)
    x = (x + 1) / 2  # scale to [0, 1]

    grid = make_grid(x, nrow=int(num_samples**0.5), padding=2)
    unet.train()
    return grid

# Training 
def train(config_path: str = 'diffusion_demo/config/train_config.toml'):
    # Load config
    config = toml.load(config_path)
    device = select_device(config)

    print(f"\n [DEVICE] Using device: {device}\n")

    # make save dir 
    exp_name = config['experiment']['name']
    save_dir = config['experiment']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # wandb
    wandb_cfg = config.get('wandb', {})
    use_wandb = wandb_cfg.get('enabled', False)

    if use_wandb:
        wandb.init(project=wandb_cfg.get('project', 'diffusion_project'),
                   entity=wandb_cfg.get('entity', None), 
                   name=wandb_cfg.get('run_name', exp_name),
                   config=config)

    # Dataloaders
    train_loader = get_dataloader(batch_size=config['training']['batch_size'], 
                                  shuffle=True, 
                                  num_workers=config['training']['num_workers'],
                                  root=config['training']['data_root'],
                                  train=True)
    
    # Scheduler
    diffusion_cfg = config['diffusion']
    scheduler = DiffusionScheduler(timesteps=diffusion_cfg['timesteps'], 
                                  beta_start=diffusion_cfg['beta_start'], 
                                  beta_end=diffusion_cfg['beta_end'],
                                  device=device)


    # Clip conditioner 
    clip_cfg = config['clip']
    clip_cond = CLIPTextConditioner(model_name=clip_cfg['model_name'],
                                    pretrained=clip_cfg['pretrained'],
                                    normalize=clip_cfg['normalize'],
                                    device=device)
    
    # UNet model
    model_cfg = config['model']
    unet = UNetConditional(in_ch=model_cfg['in_channels'],
                           base_ch=model_cfg['base_channels'],
                           time_dim=model_cfg['time_embedding_dim'],
                           cond_dim=clip_cond.cond_dim,
                           num_head=model_cfg['num_heads']).to(device)
    
    # optimizer
    optimizer = optim.Adam(unet.parameters(), lr=config['training']['learning_rate'])
    mse_loss = nn.MSELoss()

    # Training 
    log_every = config['training']['log_every']
    save_envery = config['training']['save_every']
    epochs = config['training']['epochs']


    sample_enabled = config['sampling']['enabled']
    sample_every_epoch = config['sampling']['sample_every_epoch']
    sample_num_samples = config['sampling']['num_samples']
    sample_steps = config['sampling']['steps']

    global_step = 0

    print('\n [TRAINING] Start training...\n')
    for epoch in range(1, epochs+1):
        for batch_idx, (imgs, labels, prompts) in enumerate(train_loader):
            x0 = imgs.to(device)  # [B, 3, 32, 32]
            B = x0.size(0)

            # sample timestep t
            t = torch.randint(0, scheduler.timesteps, (B,), device=device).long()  # random timesteps

            # q(x_t | x_0)
            x_t, noise = scheduler.q_sample(x0, t)

            # Encode prompt 
            cond_tokens = clip_cond(prompts).to(device)  # [B, L, cond_dim]

            # Predict noise
            pred_noise = unet(x_t, t, cond_tokens)

            # MSE loss
            loss = mse_loss(pred_noise, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 

            global_step += 1

            if batch_idx % log_every == 0:
                msg = f"Epoch [{epoch}/{epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                print(msg)
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch, "train/step": global_step})

        # save checkpoint
        if epoch % save_envery == 0:
            ckpt = {
                'epoch': epoch,
                'model': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            ckpt_path = os.path.join(save_dir, f"{exp_name}_epoch{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(f" [CHECKPOINT] Saved model checkpoint at epoch {epoch} to {ckpt_path}")

        # sampling and log images
        if sample_enabled and (epoch % sample_every_epoch == 0):
            print(f" [SAMPLING] Sampling images at epoch {epoch}...")
            grid = sample_images(
                unet = unet,
                scheduler = scheduler,
                clip_cond = clip_cond,
                device = device,
                num_samples = sample_num_samples,
                steps = sample_steps
            )

            if use_wandb:
                wandb.log(
                    {
                        'samples': wandb.Image(
                            grid, 
                            caption=f"Sampled images at epoch {epoch}"
                        )
                    },
                    step = global_step,
                )
        
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()