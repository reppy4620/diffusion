import numpy as np
import ot
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
    
    res = integrate.solve_ivp(ode_func, (0., 1.), x.reshape((-1,)).cpu().numpy(), method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x

def loss_fn(model, x_1, t):
    bs = x_1.size(0)
    x_0 = torch.randn_like(x_1)
    a = ot.unif(bs)
    b = ot.unif(bs)
    M = torch.cdist(x_0.reshape(bs, -1), x_1.reshape(bs, -1)) ** 2
    pi = ot.emd(a, b, M.detach().cpu().numpy())
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=bs)
    i, j = np.divmod(choices, pi.shape[1])
    x_0 = x_0[i]
    x_1 = x_1[j]

    x_t = t[:, None, None, None] * x_1 + (1 - t[:, None, None, None]) * x_0
    v = model(x_t, t)
    loss = F.mse_loss(x_1 - x_0, v)
    return loss

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fashion_mnist')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    dim = args.dim
    channels = args.channels
    batch_size = args.batch_size
    n_epochs = args.epoch
    save_interval = args.save_interval
    ds_name = args.dataset

    img_key = 'img' if ds_name == 'cifar10' else 'image'
    img_convert = 'RGB' if ds_name == 'cifar10' else 'L'
    image_size = 32 if ds_name == 'cifar10' else 28

    torch.manual_seed(42)

    output_dir = Path(f'../out/{ds_name}')
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    def handle_batch(batch):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(device)

        t = torch.empty(size=(batch_size,), device=device).uniform_(0., 1.)
        loss = loss_fn(model, x, t)
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
            images = sample_ode(model, image_size, channels=channels)
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
