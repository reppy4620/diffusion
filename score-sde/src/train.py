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

from model import Unet
from sde import (
    VESDE,
    VPSDE,
    SubVPSDE
)

timesteps = 300
eps = 1e-5

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
    parser.add_argument('--dataset', type=str, default='fashion_mnist')
    parser.add_argument('--sde_type', type=str, default='vp', choices=['ve', 'vp', 'subvp'])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--channels', type=int, default=1)
    args = parser.parse_args()

    image_size = 28
    dim = 32
    channels = args.channels
    batch_size = 128
    n_epochs = args.epoch
    save_interval = args.save_interval
    ds_name = args.dataset
    sde_type = args.sde_type
    img_key = 'img' if ds_name == 'cifar10' else 'image'
    img_convert = 'RGB' if ds_name == 'cifar10' else 'L'
    image_size = 32 if ds_name == 'cifar10' else 28

    torch.manual_seed(42)

    output_dir = Path(f'../out/{ds_name}/{sde_type}')
    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1)
    ])

    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert(img_convert)) for image in examples[img_key]]
        del examples[img_key]
        return examples

    ds = load_dataset(ds_name)
    transformed_ds = ds.with_transform(transforms).remove_columns("label")

    # create dataloader
    dl = DataLoader(transformed_ds["train"], batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(dim=dim, channels=channels, dim_mults=(1, 2, 4)).to(device)
    if sde_type == 've':
        sde = VESDE()
    elif sde_type == 'vp':
        sde = VPSDE()
    elif sde_type == 'subvp':
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
    for epoch in range(1, n_epochs + 1):
        losses = list()
        bar = tqdm(dl, total=len(dl), desc=f'Epoch {epoch}: ')
        model.train()
        for batch in bar:
            optimizer.zero_grad()
            loss = handle_batch(batch)
            loss.backward()

            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix_str(f'Loss: {np.mean(losses):.6f}')
        train_losses.append(np.mean(losses))
        if epoch % save_interval == 0:
            model.eval()
            images = sample_ode(sde, model, image_size, channels=channels)
            img = make_grid(images, nrow=4, normalize=True)
            img = T.ToPILImage()(img)
            img.save(img_dir / f'epoch_{epoch}.png')

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_dir / f'epoch_{epoch:05d}.pth')


if __name__ == '__main__':
    main()
