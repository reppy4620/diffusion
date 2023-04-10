import math
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
from argparse import ArgumentParser
from omegaconf import OmegaConf

from model import ConsistencyModel

def t_schedule(rho, eps, N, T):
    # paper p.4
    return torch.tensor([
        ( eps ** (1 / rho) + (i - 1) / (N - 1) * (T ** (1 / rho) - eps ** (1 / rho)) ) ** rho
        for i in range(N)
    ])

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
    dl = DataLoader(transformed_ds["train"], batch_size=config.batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConsistencyModel(**config.model).to(device)
    ema_model = ConsistencyModel(**config.model).to(device)
    ema_model.load_state_dict(model.state_dict())

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_losses = list()
    for epoch in range(1, config.epochs + 1):
        losses = list()
        bar = tqdm(dl, total=len(dl), desc=f'Epoch {epoch}: ')
        model.train()
        for i, batch in enumerate(bar):
            optimizer.zero_grad()
            
            batch_size = batch["pixel_values"].shape[0]
            x = batch["pixel_values"].to(device)

            # paper p.25
            N = math.ceil(
                math.sqrt(
                    (epoch * len(dl) + i) / (config.epochs * len(dl)) * ((config.schedule.s_1 + 1) ** 2 - config.schedule.s_0 ** 2) 
                    + config.schedule.s_0 ** 2
                )
            ) + 1
            t_boundaries = t_schedule(**config.schedule.fun, N=N).to(device)
            t_indices = torch.randint(low=0, high=N-1, size=(batch_size,))
            t1 = t_boundaries[t_indices + 1]
            t2 = t_boundaries[t_indices]
            z = torch.randn_like(x)
            loss = model.loss(x, z, t1, t2, ema_model)

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(config.schedule.mu_0) / N)
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            losses.append(loss.item())
            bar.set_postfix_str(f'N: {N}, Loss: {np.mean(losses):.6f}')
        train_losses.append(np.mean(losses))
        if epoch % config.image_interval == 0:
            model.eval()
            x = torch.randn(16, config.model.channels, config.img_size, config.img_size, device=device) * config.schedule.fun.T
            images = model.sample(
                x,
                ts=[80.0, 20.0, 5.0, 1.0]
            )
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
