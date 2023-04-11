import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from pathlib import Path
from scipy import integrate
from argparse import ArgumentParser
from omegaconf import OmegaConf

from model import Unet
from sde import (
    VESDE,
    VPSDE,
    SubVPSDE
)

eps = 1e-3

@torch.no_grad()
def p_sample(sde, model, x, t, dt):
    f, g = sde.sde(x, t)
    score = model(x, t)
    drift = (f - (g ** 2)[:, None, None, None] * score)
    diffusion =  g[:, None, None, None]
    w = torch.randn_like(x) * torch.sqrt(dt)
    x = x + drift * (-dt) + diffusion * w
    return x

@torch.no_grad()
def p_sample_ode(sde, model, x, t, dt):
    f, g = sde.sde(x, t)
    score = model(x, t)
    x = x + (f - 0.5 * (g ** 2)[:, None, None, None] * score) * (-dt)
    return x

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(sde, model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    t_1 = torch.ones((b,), device=device)
    _, std = sde.marginal_prob(torch.zeros(shape, device=device), t_1)
    x = torch.randn(shape, device=device) * std[:, None, None, None]
    ts = torch.linspace(1, eps, timesteps)
    dt = ts[0] - ts[1]

    for i in tqdm(reversed(ts), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        x = p_sample(sde, model, x, t, dt)
    return x

@torch.no_grad()
def sample(sde, model, image_size, batch_size=16, channels=3):
    return p_sample_loop(sde, model, shape=(batch_size, channels, image_size, image_size))

@torch.no_grad()
def sample_ode(sde, model, image_size, batch_size=16, channels=1):
    shape = (batch_size, channels, image_size, image_size)
    device = next(model.parameters()).device

    b = shape[0]
    t_1 = torch.ones((b,), device=device)
    _, std = sde.marginal_prob(torch.zeros(shape, device=device), t_1)
    x = torch.randn(shape, device=device) * std[:, None, None, None]
    
    def ode_func(t, x):
        x = torch.tensor(x, device=device, dtype=torch.float).reshape(shape)
        t = torch.full(size=(b,), fill_value=t, device=device, dtype=torch.float).reshape((b,))
        score = model(x, t)
        drift, _ = sde.probability_flow(score, x, t)
        return drift.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    res = integrate.solve_ivp(ode_func, (1., eps), x.reshape((-1,)).cpu().numpy(), rtol=1e-5, atol=1e-5, method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x

# forward diffusion (using the nice property)
def q_sample(sde, x_0, t, noise):        
    mean, std = sde.marginal_prob(x_0, t)
    perturb_x = mean + std[:, None, None, None] * noise
    return perturb_x, mean, std

def p_losses(sde, model, x_0, t):
    z = torch.randn_like(x_0)
    perturb_x, _, std = q_sample(sde=sde, x_0=x_0, t=t, noise=z)
    score = model(perturb_x, t)
    loss = torch.mean((score * std[:, None, None, None] + z) ** 2)
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
    
    if config.sde_type == 've':
        sde = VESDE()
    elif config.sde_type == 'vp':
        sde = VPSDE()
    elif config.sde_type == 'subvp':
        sde = SubVPSDE()
    else:
        raise NotImplementedError()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def handle_batch(batch):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(device)

        t = torch.empty(size=(batch_size,), device=device).uniform_(eps, 1)
        loss = p_losses(sde, model, x, t)
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
            images = sample_ode(model, config.image_size, channels=config.model.channels)
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
