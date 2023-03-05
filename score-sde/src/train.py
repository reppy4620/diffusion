import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from model import Unet
from sde import VPSDE

timesteps = 300
eps = 1e-5

# forward diffusion (using the nice property)
def q_sample(sde, x_0, t, noise):        
    mean, std = sde.marginal_prob(x_0, t)
    perturb_x = mean + std[:, None, None, None] * noise
    return perturb_x, mean, std

@torch.no_grad()
def p_sample(sde, model, x, t, dt):
    f, g = sde.sde(x, t)
    score = model(x, t)
    drift = (f - (g ** 2)[:, None, None, None] * score)
    diffusion =  g[:, None, None, None]
    w = torch.randn_like(x) * torch.sqrt(dt)
    return drift * dt + diffusion * w

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(sde, model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    x = torch.randn(shape, device=device)
    ts = torch.linspace(1, eps, timesteps)
    dt = ts[0] - ts[1]

    for i in tqdm(reversed(ts), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        x = p_sample(sde, model, x, t, dt)
    return x

@torch.no_grad()
def sample(sde, model, image_size, batch_size=16, channels=3):
    return p_sample_loop(sde, model, shape=(batch_size, channels, image_size, image_size))

def p_losses(sde, model, x_0, t):
    z = torch.randn_like(x_0)

    perturb_x, _, std = q_sample(sde=sde, x_0=x_0, t=t, noise=z)
    score = model(perturb_x, t)
    loss = torch.mean((score + z / std[:, None, None, None]) ** 2)
    return loss

def main():
    image_size = 28
    dim = 32
    channels = 1
    batch_size = 128
    n_epochs = 10
    save_interval = 5

    output_dir = Path('../out')
    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # define image transformations (e.g. using torchvision)
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1)
    ])

    # define function
    def transforms(examples):
        examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

    ds = load_dataset("fashion_mnist")
    transformed_ds = ds.with_transform(transforms).remove_columns("label")

    # create dataloader
    dl = DataLoader(transformed_ds["train"], batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(dim=dim, channels=channels, dim_mults=(1, 2, 4)).to(device)
    sde = VPSDE()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

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
        for batch in bar:
            optimizer.zero_grad()
            loss = handle_batch(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix_str(f'Loss: {np.mean(losses):.6f}')
        train_losses.append(np.mean(losses))
        if epoch % save_interval == 0:
            images = sample(sde, model, image_size, channels=channels)
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
