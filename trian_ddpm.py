import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1] for saving."""
    return (x.clamp(-1, 1) + 1) / 2


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from a 1D schedule tensor at batch timesteps.

    a: [T]
    t: [B]
    return: [B, 1, 1, 1] for broadcasting over image tensors
    """
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    """
    Closed-form forward noising:
        x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
    """
    sqrt_ab = extract(schedule.sqrt_alpha_bars, t, x0.shape)
    sqrt_omab = extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape)
    return sqrt_ab * x0 + sqrt_omab * noise


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """
    A small time-conditioned U-Net for MNIST noise prediction.
    Input:  x_t with shape [B, 1, 28, 28]
    Output: predicted epsilon with shape [B, 1, 28, 28]
    """
    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        self.down0 = ResBlock(32, 32, time_dim)      # 28 x 28
        self.down1 = ResBlock(32, 64, time_dim)      # 14 x 14
        self.down2 = ResBlock(64, 128, time_dim)     # 7 x 7

        self.mid = ResBlock(128, 128, time_dim)      # 7 x 7

        self.up1 = ResBlock(128 + 64, 64, time_dim)  # 14 x 14
        self.up2 = ResBlock(64 + 32, 32, time_dim)   # 28 x 28

        self.out_norm = group_norm(32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(t)

        x = self.in_conv(x)
        h0 = self.down0(x, temb)  # [B, 32, 28, 28]

        h = F.avg_pool2d(h0, kernel_size=2)  # [B, 32, 14, 14]
        h1 = self.down1(h, temb)             # [B, 64, 14, 14]

        h = F.avg_pool2d(h1, kernel_size=2)  # [B, 64, 7, 7]
        h2 = self.down2(h, temb)             # [B, 128, 7, 7]

        h = self.mid(h2, temb)

        h = F.interpolate(h, size=h1.shape[-2:], mode="nearest")
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h, temb)

        h = F.interpolate(h, size=h0.shape[-2:], mode="nearest")
        h = torch.cat([h, h0], dim=1)
        h = self.up2(h, temb)

        return self.out_conv(F.silu(self.out_norm(h)))


@torch.no_grad()
def p_sample(model: nn.Module, x: torch.Tensor, t_idx: int, schedule: DiffusionSchedule) -> torch.Tensor:
    """
    One DDPM reverse sampling step:
        mu = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
        x_{t-1} = mu + sqrt(beta_t) z
    """
    b = x.shape[0]
    t = torch.full((b,), t_idx, device=x.device, dtype=torch.long)

    beta_t = extract(schedule.betas, t, x.shape)
    alpha_t = extract(schedule.alphas, t, x.shape)
    alpha_bar_t = extract(schedule.alpha_bars, t, x.shape)

    pred_noise = model(x, t)
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise
    )

    if t_idx == 0:
        return mean

    z = torch.randn_like(x)
    return mean + torch.sqrt(beta_t) * z


@torch.no_grad()
def sample(model: nn.Module, n: int, schedule: DiffusionSchedule, device: torch.device, save_steps=None):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)

    trajectory = []
    if save_steps is not None:
        trajectory.append((schedule.timesteps, x.detach().cpu()))  # pure noise x_T

    for t_idx in reversed(range(schedule.timesteps)):
        x = p_sample(model, x, t_idx, schedule)
        if save_steps is not None and t_idx in save_steps:
            trajectory.append((t_idx, x.detach().cpu()))

    return x.detach().cpu(), trajectory


@torch.no_grad()
def save_forward_noising_grid(loader, schedule: DiffusionSchedule, device: torch.device, out_dir: str):
    x0, _ = next(iter(loader))
    x0 = x0[:1].to(device)
    fixed_noise = torch.randn_like(x0)

    times = [0, schedule.timesteps // 4, schedule.timesteps // 2, 3 * schedule.timesteps // 4, schedule.timesteps - 1]
    imgs = []
    for t_idx in times:
        t = torch.tensor([t_idx], device=device, dtype=torch.long)
        xt = q_sample(x0, t, fixed_noise, schedule)
        imgs.append(xt.cpu())

    imgs = torch.cat(imgs, dim=0)
    save_image(denorm(imgs), os.path.join(out_dir, "forward_noising_grid.png"), nrow=len(times))


@torch.no_grad()
def save_reverse_trajectory(model: nn.Module, schedule: DiffusionSchedule, device: torch.device, out_dir: str):
    save_steps = [3 * schedule.timesteps // 4, schedule.timesteps // 2, schedule.timesteps // 4, 0]
    _, trajectory = sample(model, n=8, schedule=schedule, device=device, save_steps=save_steps)

    rows = []
    for _, imgs in trajectory:
        rows.append(denorm(imgs))
    grid_imgs = torch.cat(rows, dim=0)
    save_image(grid_imgs, os.path.join(out_dir, "reverse_denoising_trajectory.png"), nrow=8)


@torch.no_grad()
def save_generated_digits(model: nn.Module, schedule: DiffusionSchedule, device: torch.device, out_dir: str):
    samples, _ = sample(model, n=64, schedule=schedule, device=device, save_steps=None)
    save_image(denorm(samples), os.path.join(out_dir, "generated_digits.png"), nrow=8)


def save_loss_curve(loss_history, out_dir: str):
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("DDPM Training Loss on MNIST")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    schedule = DiffusionSchedule(args.timesteps, device)
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x0, _ in train_loader:
            x0 = x0.to(device)
            b = x0.shape[0]

            t = torch.randint(0, schedule.timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, schedule)

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1:03d}/{args.epochs}, loss = {avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timesteps": args.timesteps,
            "loss_history": loss_history,
        },
        os.path.join(args.out_dir, "mnist_ddpm_checkpoint.pt"),
    )

    save_forward_noising_grid(train_loader, schedule, device, args.out_dir)
    save_reverse_trajectory(model, schedule, device, args.out_dir)
    save_generated_digits(model, schedule, device, args.out_dir)
    save_loss_curve(loss_history, args.out_dir)

    print("Saved outputs:")
    print(f"  {args.out_dir}/forward_noising_grid.png")
    print(f"  {args.out_dir}/reverse_denoising_trajectory.png")
    print(f"  {args.out_dir}/generated_digits.png")
    print(f"  {args.out_dir}/loss_curve.png")
    print(f"  {args.out_dir}/mnist_ddpm_checkpoint.pt")


if __name__ == "__main__":
    main()
