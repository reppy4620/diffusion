import numpy as np
import torch
import torch.nn.functional as F
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

eps = 1e-3

@torch.no_grad()
def sample_ode(model, image_size, batch_size=16, channels=1):
    shape = (batch_size, channels, image_size, image_size)
    device = next(model.parameters()).device

    b = shape[0]
    x = torch.randn(shape, device=device)
    
    def ode_func(t, x):
        x = torch.tensor(x, device=device, dtype=torch.float).reshape(shape)
        t = torch.full(size=(b,), fill_value=t, device=device, dtype=torch.float).reshape((b,))
        v = model(x, t)
        return v.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    res = integrate.solve_ivp(ode_func, (eps, 1.), x.reshape((-1,)).cpu().numpy(), method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x

def loss_fn(model, x_1, t):
    x_0 = torch.randn_like(x_1)
    x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0
    v = model(x_t, t)
    loss = F.mse_loss(x_1 - x_0, v)
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    def handle_batch(batch):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(device)

        t = torch.empty(size=(batch_size,), device=device).uniform_(eps, 1)
        loss = loss_fn(model, x, t)
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
