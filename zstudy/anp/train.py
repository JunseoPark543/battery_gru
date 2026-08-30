import os
from pathlib import Path

import torch as t
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import LatentModel
from preprocess import collate_fn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoint"

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # MNIST is intentionally kept local. The repository .gitignore excludes
    # data/, and download=False prevents this training script from performing
    # an unexpected network download on another machine.
    try:
        train_dataset = torchvision.datasets.MNIST(
            root=DATA_ROOT,
            train=True,
            download=False,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"MNIST was not found under {DATA_ROOT / 'MNIST'}. "
            "Download it once on this server before starting training."
        ) from exc

    epochs = 200
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = LatentModel(128).to(device)
    model.train()

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    optim = t.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs" / "zstudy_anp_mnist"))
    global_step = 0
    try:
        for epoch in range(epochs):
            dloader = DataLoader(
                train_dataset,
                batch_size=16,
                collate_fn=collate_fn,
                shuffle=True,
                num_workers=min(16, os.cpu_count() or 1),
                pin_memory=device.type == "cuda",
            )
            pbar = tqdm(dloader, desc=f"epoch {epoch + 1}/{epochs}")
            for data in pbar:
                global_step += 1
                adjust_learning_rate(optim, global_step)
                context_x, context_y, target_x, target_y = (
                    tensor.to(device, non_blocking=True) for tensor in data
                )

                # Pass through the latent model.
                _, kl, loss = model(
                    context_x, context_y, target_x, target_y
                )

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                writer.add_scalars(
                    "training_loss",
                    {
                        "loss": float(loss.detach()),
                        "kl": float(kl.detach().mean()),
                    },
                    global_step,
                )
                pbar.set_postfix(loss=f"{float(loss.detach()):.6f}")

            # Save one resumable checkpoint after every epoch.
            t.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                },
                CHECKPOINT_ROOT / f"checkpoint_{epoch + 1}.pth.tar",
            )
    finally:
        writer.close()


if __name__ == '__main__':
    main()
