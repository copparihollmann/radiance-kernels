#!/usr/bin/env python3
"""Make small, reproducible trained checkpoints for SpectFormer inference tests.

This is a functional validation fixture, not an ImageNet accuracy experiment.
The 1,000-output classification heads retain the upstream inference shapes;
Handwritten-digit labels use indices 0 through 9.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from hierarchical_prepare import SOURCE_SHA256, load_source as load_hierarchical
from prepare import UPSTREAM_SOURCE_SHA256, load_upstream, sha256


def train(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms
    from PIL import Image
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    args.out.mkdir(parents=True, exist_ok=True)
    source_hash = (SOURCE_SHA256 if args.variant == "h-s" else
                   UPSTREAM_SOURCE_SHA256)
    if sha256(args.upstream_source) != source_hash:
        raise ValueError("source is not the pinned upstream implementation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.variant == "xs":
        upstream = load_upstream(args.upstream_source)
        model = upstream.SpectFormer(img_size=224, patch_size=16,
                                     embed_dim=384, depth=12,
                                     num_classes=1000, drop_rate=0.0,
                                     drop_path_rate=0.0)
    else:
        upstream = load_hierarchical(args.upstream_source)
        model = upstream.spectformer_s(pretrained=False, num_classes=1000,
                                       token_label=True, drop_path_rate=0.0)
        model.mix_token = False  # avoid training-only token mixing; eval graph is unchanged
    model.to(device).train()
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    if args.dataset == "digits":
        with gzip.open(args.data_root, "rt") as stream:
            matrix = np.loadtxt(stream, delimiter=",", dtype=np.float32)
        raw_images = matrix[:, :64].reshape(-1, 8, 8)
        digit_labels = matrix[:, 64].astype(np.int64).tolist()

        class DigitDataset(torch.utils.data.Dataset):
            targets = digit_labels

            def __len__(self):
                return len(digit_labels)

            def __getitem__(self, index):
                pixels = np.uint8(np.rint(raw_images[index] * (255.0 / 16.0)))
                image = Image.fromarray(pixels, mode="L").convert("RGB")
                return transform(image), digit_labels[index]

        dataset = DigitDataset()
        dataset_name = "scikit-learn 1.6.1 handwritten digits, 8x8"
        dataset_sha256 = sha256(args.data_root)
    else:
        dataset = datasets.CIFAR10(root=args.data_root, train=True,
                                   download=True, transform=transform)
        dataset_name = "torchvision CIFAR10 train split"
        dataset_sha256 = None
    selected: list[int] = []
    held_out: list[int] = []
    counts = [0] * 10
    for index, label in enumerate(dataset.targets):
        if counts[label] < args.samples_per_class:
            selected.append(index)
        elif counts[label] == args.samples_per_class:
            held_out.append(index)
        counts[label] += 1
        if all(count > args.samples_per_class for count in counts):
            break
    if len(selected) != 10 * args.samples_per_class or len(held_out) != 10:
        raise ValueError("ten-class training subset is incomplete")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selected), batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    for pixels, labels in loader:
        pixels, labels = pixels.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(pixels)
        if args.variant == "h-s":
            class_logits, auxiliary_logits, _ = output
            loss = (F.cross_entropy(class_logits, labels) +
                    0.5 * F.cross_entropy(auxiliary_logits.mean(dim=1), labels))
        else:
            loss = F.cross_entropy(output, labels)
        if not torch.isfinite(loss):
            raise ValueError("training produced a nonfinite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    checkpoint = args.out / "checkpoint.pth"
    torch.save({"model": model.cpu().state_dict(),
                "training_steps": len(losses), "variant": args.variant}, checkpoint)
    examples = np.stack([dataset[index][0].numpy() for index in held_out[:2]], axis=0)
    for index, values in enumerate(examples):
        np.save(args.out / f"heldout-{index}.npy", values[None].astype(np.float32))
    record = {
        "purpose": "trained_weight_functional_inference_fixture",
        "variant": args.variant, "source_sha256": source_hash,
        "dataset": dataset_name, "dataset_sha256": dataset_sha256,
        "dataset_subset_indices": selected,
        "heldout_indices": held_out[:2],
        "samples_per_class": args.samples_per_class,
        "label_to_output_index": "digit class index 0..9 in 1000-way head",
        "preprocessing": "bicubic resize to 224x224, ImageNet mean/std",
        "seed": args.seed, "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "steps": len(losses),
        "loss_first": losses[0], "loss_last": losses[-1],
        "checkpoint_sha256": sha256(checkpoint),
        "device": str(device),
        "imagenet_accuracy_claim": False,
    }
    (args.out / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("xs", "h-s"), required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("digits", "cifar10"), default="digits")
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1827)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
