import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

try:
    from torch.func import jvp
except Exception:
    from functorch import jvp

from src.ot_mnist.metrics import classifier_metrics
from src.ot_mnist.models import EMA, MNISTClassifier, SmallUNet
from src.ot_mnist.ot import hard_assignment_pair, sinkhorn_hard_pair


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["meanflow", "ot_gmf_hard", "ot_gmf_sinkhorn_hard"], required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./outputs")
    p.add_argument("--classifier", default="./outputs/mnist_classifier.pt")
    p.add_argument("--resume", default="")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=256, help="Global batch size across all GPUs.")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-n", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--r-not-t-ratio", type=float, default=0.25)
    p.add_argument("--sinkhorn-eps", type=float, default=0.05)
    p.add_argument("--sinkhorn-iters", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--adaptive-p", type=float, default=1.0)
    return p.parse_args()


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank
    return False, 0, 1, 0


def is_main(rank):
    return rank == 0


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def sample_r_t(b, device, ratio):
    a = torch.rand(b, device=device)
    c = torch.rand(b, device=device)
    r = torch.minimum(a, c)
    t = torch.maximum(a, c)
    same = torch.rand(b, device=device) > ratio
    r = torch.where(same, t, r)
    return r, t


def adaptive_loss(error, p=1.0, c=1e-3):
    per = error.pow(2).flatten(1).sum(dim=1)
    if p == 0:
        return per.mean()
    w = (per + c).pow(-p).detach()
    return (w * per).mean()


def make_loader(args, ddp, rank, world):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if ddp and not is_main(rank):
        dist.barrier()
    ds = datasets.MNIST(args.data_dir, train=True, download=is_main(rank), transform=tfm)
    if ddp and is_main(rank):
        dist.barrier()
    if ddp:
        dist.barrier()
    per_rank_batch = max(1, args.batch_size // world)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        ds,
        batch_size=per_rank_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def infinite_loader(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def model_forward(model, z, r, t):
    return model(z, r, t)


def train_step(args, model, opt, batch, device):
    x = batch[0].to(device, non_blocking=True)
    e = torch.randn_like(x)
    b = x.shape[0]

    if args.method == "meanflow":
        r, t = sample_r_t(b, device, args.r_not_t_ratio)
        z = (1.0 - t[:, None, None, None]) * x + t[:, None, None, None] * e
        v = e - x

        def fn(z_, r_, t_):
            return model_forward(model, z_, r_, t_)

        u, dudt = jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        u_tgt = (v - (t - r)[:, None, None, None] * dudt).detach()
        loss = adaptive_loss(u - u_tgt, p=args.adaptive_p)
    else:
        if args.method == "ot_gmf_hard":
            x_pair, _ = hard_assignment_pair(e.detach(), x.detach())
        else:
            x_pair, _ = sinkhorn_hard_pair(
                e.detach(), x.detach(), epsilon=args.sinkhorn_eps, iters=args.sinkhorn_iters
            )
        r, t = sample_r_t(b, device, 1.0)
        z = (1.0 - t[:, None, None, None]) * x_pair + t[:, None, None, None] * e
        target = (e - x_pair).detach()
        u = model(z, r, t)
        loss = adaptive_loss(u - target, p=args.adaptive_p)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach()


@torch.no_grad()
def generate(model, n, device, batch_size=512):
    model.eval()
    outs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        e = torch.randn(b, 1, 28, 28, device=device)
        r = torch.zeros(b, device=device)
        t = torch.ones(b, device=device)
        x = e - model(e, r, t)
        outs.append(x.clamp(-1, 1).cpu())
    model.train()
    return torch.cat(outs, dim=0)


@torch.no_grad()
def evaluate(args, model, ema, device, step, out_dir):
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    backup = ema.copy_to(unwrap(model)) if ema is not None else None
    samples = generate(unwrap(model), args.eval_n, device)
    grid = make_grid((samples[:100] + 1) / 2, nrow=10)
    save_image(grid, eval_dir / f"{args.method}_step_{step:07d}.png")

    metrics = {}
    clf_path = Path(args.classifier)
    if clf_path.exists():
        classifier = MNISTClassifier().to(device)
        ckpt = torch.load(clf_path, map_location=device)
        classifier.load_state_dict(ckpt["model"])
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_ds = datasets.MNIST(args.data_dir, train=False, download=False, transform=tfm)
        idx = torch.randperm(len(test_ds))[: args.eval_n].tolist()
        real = torch.stack([test_ds[i][0] for i in idx], dim=0)
        metrics.update(classifier_metrics(classifier, samples, real, device))
    if backup is not None:
        ema.restore(unwrap(model), backup)
    return metrics


def save_ckpt(path, args, model, ema, opt, step, images_seen, logs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "step": step,
            "images_seen": images_seen,
            "model": unwrap(model).state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": opt.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_random": random.getstate(),
            "logs": logs,
        },
        path,
    )


def load_ckpt(path, model, ema, opt, device):
    ckpt = torch.load(path, map_location=device)
    unwrap(model).load_state_dict(ckpt["model"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    opt.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng_torch"])
    if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_random"])
    return ckpt["step"], ckpt.get("images_seen", 0), ckpt.get("logs", [])


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    ddp, rank, world, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    out_dir = Path(args.out_dir) / args.method
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(vars(args), indent=2))
        print(f"world_size={world} device={device}")

    loader, sampler = make_loader(args, ddp, rank, world)
    batches = infinite_loader(loader, sampler)

    model = SmallUNet(base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(unwrap(model).parameters(), lr=args.lr, weight_decay=0.0)
    ema = EMA(unwrap(model), decay=args.ema)

    start_step = 0
    images_seen = 0
    logs = []
    resume_path = args.resume or str(out_dir / "checkpoints" / "latest.pt")
    if Path(resume_path).exists():
        start_step, images_seen, logs = load_ckpt(resume_path, model, ema, opt, device)
        if is_main(rank):
            print(f"resumed from {resume_path} at step {start_step}")

    t0 = time.time()
    rolling = []
    for step in range(start_step + 1, args.steps + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        s0 = time.time()
        loss = train_step(args, model, opt, next(batches), device)
        ema.update(unwrap(model))
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            peak_mb = 0.0
        step_sec = time.time() - s0
        images_seen += args.batch_size
        rolling.append((loss.item(), step_sec, peak_mb))

        if is_main(rank) and (step % args.log_every == 0 or step == 1):
            avg_loss = float(np.mean([x[0] for x in rolling]))
            avg_sec = float(np.mean([x[1] for x in rolling]))
            avg_mem = float(np.mean([x[2] for x in rolling]))
            row = {
                "method": args.method,
                "step": step,
                "images_seen": images_seen,
                "elapsed_min": (time.time() - t0) / 60,
                "loss": avg_loss,
                "sec_per_step": avg_sec,
                "peak_mem_mb": avg_mem,
            }
            print(row)
            append_csv(out_dir / "logs.csv", row)
            logs.append(row)
            rolling.clear()

        if is_main(rank) and step % args.eval_every == 0:
            metrics = evaluate(args, model, ema, device, step, out_dir)
            row = {
                "method": args.method,
                "step": step,
                "images_seen": images_seen,
                "elapsed_min": (time.time() - t0) / 60,
                **metrics,
            }
            print("eval", row)
            append_csv(out_dir / "eval.csv", row)

        if is_main(rank) and step % args.save_every == 0:
            ckpt_dir = out_dir / "checkpoints"
            save_ckpt(ckpt_dir / "latest.pt", args, model, ema, opt, step, images_seen, logs)
            if step % (args.save_every * 10) == 0:
                save_ckpt(ckpt_dir / f"step_{step:07d}.pt", args, model, ema, opt, step, images_seen, logs)

    if is_main(rank):
        save_ckpt(out_dir / "checkpoints" / "latest.pt", args, model, ema, opt, args.steps, images_seen, logs)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
