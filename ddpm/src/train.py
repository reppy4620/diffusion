import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf

from schedule import linear_beta_schedule
from model import Unet

timesteps = 1000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)
    return img

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    torch.manual_seed(42)

    output_dir = Path(f'{config.output_dir}/{config.dataset}')
    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    if config.dataset.startswith('huggan'):
        transform = T.Compose([
            T.Resize((config.img_size, config.img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])
    else:
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])

    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert(config.img_convert)) for image in examples[config.img_key]]
        del examples[config.img_key]

        return examples

    ds = load_dataset(config.dataset)
    if hasattr(config, 'filter'):
        if config.filter == 'cat':
            ds = ds.filter(lambda x: x["label"] == 0)
    transformed_ds = ds.with_transform(transforms).remove_columns("label")

    # create dataloader
    dl = DataLoader(transformed_ds["train"], batch_size=config.batch_size, shuffle=True, num_workers=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(**config.model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    def handle_batch(batch):
        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = p_losses(model, batch, t, loss_type='l2')
        return loss

    train_losses = list()
    for epoch in range(1, config.epochs + 1):
        losses = list()
        bar = tqdm(dl, total=len(dl), desc=f'Epoch {epoch}: ')
        for batch in bar:
            optimizer.zero_grad()
            loss = handle_batch(batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix_str(f'Loss: {np.mean(losses):.6f}')
        train_losses.append(np.mean(losses))
        if epoch % config.image_interval == 0:
            images = sample(model, config.image_size, channels=config.model.channels)
            img = make_grid(images, nrow=4, normalize=True)
            img = T.ToPILImage()(img)
            img.save(img_dir / f'epoch_{epoch}.png')

        if epoch % config.ckpt_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_dir / f'epoch_{epoch:05d}.pth')


if __name__ == '__main__':
    main()
