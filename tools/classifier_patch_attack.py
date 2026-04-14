"""Adversarial Patch attack for classification models using ImageNet dataset."""

import argparse
import os
import os.path as osp
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(__file__)))
from asd_defense import DWTPreprocessPlugin


class PatchTransformer(nn.Module):
    """Patch Transformer for classification models.

    Applies random transformations to the patch and places it at random positions.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 64,
        contrast: tuple[float, float] = (0.8, 1.2),
        brightness: tuple[float, float] = (-0.1, 0.1),
        angle: tuple[float, float] = (-20.0, 20.0),
        noise_factor: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_creator_size = patch_size * 3  # Create patch with 3x pixel size
        self.min_contrast, self.max_contrast = contrast
        self.min_brightness, self.max_brightness = brightness
        self.min_angle, self.max_angle = angle
        self.noise_factor = noise_factor

        # Create circular mask (will be resized when placing)
        self.circular_mask = self._create_circular_mask()

    def _create_circular_mask(self):
        """Create a circular mask for the patch."""
        mask = torch.zeros(1, 1, self.patch_creator_size, self.patch_creator_size)
        center = self.patch_creator_size // 2
        radius = self.patch_creator_size // 2

        for i in range(self.patch_creator_size):
            for j in range(self.patch_creator_size):
                if (i - center) ** 2 + (j - center) ** 2 <= radius**2:
                    mask[0, 0, i, j] = 1.0

        return mask

    def forward(self, patch: torch.Tensor, batch_size: int):
        """Transform the patch and place it at random positions."""
        # Expand patch to batch size
        patch = patch.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Apply circular mask
        mask = self.circular_mask.to(patch.device).expand(batch_size, 3, -1, -1)
        patch = patch * mask

        # Apply contrast, brightness, and noise
        contrast = torch.empty(batch_size, 1, 1, 1, device=patch.device)
        contrast.uniform_(self.min_contrast, self.max_contrast)

        brightness = torch.empty(batch_size, 1, 1, 1, device=patch.device)
        brightness.uniform_(self.min_brightness, self.max_brightness)

        noise = torch.empty_like(patch, device=patch.device)
        noise.uniform_(-self.noise_factor, self.noise_factor)

        patch = patch * contrast + brightness + noise
        patch = torch.clamp(patch, 0, 1)

        # Apply random rotation
        angle = torch.empty(batch_size, device=patch.device)
        angle.uniform_(self.min_angle, self.max_angle)
        angle_rad = angle * torch.pi / 180

        # Calculate rotation matrices
        cos = torch.cos(angle_rad)
        sin = torch.sin(angle_rad)

        theta = torch.zeros(batch_size, 2, 3, device=patch.device)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos

        # Rotate patch
        grid = nn.functional.affine_grid(theta, patch.size(), align_corners=False)
        patch = nn.functional.grid_sample(patch, grid, align_corners=False)
        mask = nn.functional.grid_sample(mask, grid, align_corners=False)

        # Resize patch to the desired patch size
        patch = nn.functional.interpolate(
            patch,
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        )
        mask = nn.functional.interpolate(
            mask,
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        )

        # Place patch at random positions
        tx = torch.empty(batch_size, device=patch.device, dtype=torch.long)
        ty = torch.empty(batch_size, device=patch.device, dtype=torch.long)

        # Calculate valid range for patch placement
        max_offset = self.img_size - self.patch_size
        tx.random_(0, max_offset + 1)
        ty.random_(0, max_offset + 1)

        # Create empty patch and mask tensors of image size
        patch_resized = torch.zeros(
            batch_size, 3, self.img_size, self.img_size, device=patch.device
        )
        mask_resized = torch.zeros(
            batch_size, 3, self.img_size, self.img_size, device=patch.device
        )

        # Place each patch at its random position
        for i in range(batch_size):
            x = tx[i]
            y = ty[i]
            patch_resized[i, :, y : y + self.patch_size, x : x + self.patch_size] = (
                patch[i]
            )
            mask_resized[i, :, y : y + self.patch_size, x : x + self.patch_size] = mask[
                i
            ]

        return patch_resized, mask_resized


def load_imagenet(data_dir: str, batch_size: int = 32):
    """Load ImageNet dataset."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    dataset = Subset(
        dataset, random.sample(range(len(dataset)), int(len(dataset) * 0.1))
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def train_patch(args):
    """Train adversarial patch for classification models."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.weight:
        model = resnet50()
        model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(device)
    model.eval().requires_grad_(False)

    # Load dataset
    dataloader = load_imagenet(args.data_dir, args.batch_size)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Initialize patch with 3x size
    patch = torch.full(
        (3, args.patch_size * 3, args.patch_size * 3),
        0.5,
        device=device,
        requires_grad=True,
    )

    # Initialize patch transformer
    transformer = PatchTransformer(
        img_size=224,
        patch_size=args.patch_size,
        contrast=(0.9, 1.1),
        brightness=(-0.1, 0.1),
        angle=(-20, 20),
        noise_factor=0.05,
    )
    transformer = transformer.to(device)

    if args.defense:
        dwt_pool = dict(
            type="AvgPool2d",
            kernel_size=3,
            stride=1,
            padding=1,
        )
        dwt_preprocess = DWTPreprocessPlugin(
            max_level=5,
            threshold=0.17,
            pool=dwt_pool,
        )

    # Initialize optimizer
    optimizer = optim.Adam([patch], lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Target class (randomly selected)
    target_class = args.target_class
    print(f"Target class: {target_class}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Generate transformed patch
            patch_transformed, mask = transformer(patch, images.shape[0])

            # Apply patch to images
            adv_images = images * (1 - mask) + patch_transformed
            adv_images = torch.clamp(adv_images, 0, 1)

            if args.defense:
                adv_images = dwt_preprocess(adv_images)

            # Forward pass
            outputs = model(normalize(adv_images))

            # Calculate loss (targeted attack)
            loss = torch.nn.functional.cross_entropy(
                outputs, torch.full_like(labels, target_class)
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update statistics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        # Update learning rate
        scheduler.step()

        # Calculate epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch + 1}/{args.epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%"
        )

        # Save patch
        if (epoch + 1) % args.save_interval == 0:
            patch_save = patch.detach().cpu()
            torchvision.utils.save_image(
                patch_save, osp.join(args.output_dir, f"patch_epoch_{epoch + 1}.png")
            )
            print(f"Saved patch at epoch {epoch + 1}")

    # Save final patch
    patch_save = patch.detach().cpu()
    torchvision.utils.save_image(
        patch_save, osp.join(args.output_dir, "patch_final.png")
    )
    print("Saved final patch")


def evaluate_patch(args):
    """Evaluate adversarial patch on classification models."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.weight:
        model = resnet50()
        model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(device)
    model.eval().requires_grad_(False)

    # Load dataset
    dataloader = load_imagenet(args.data_dir, args.batch_size)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Load patch
    patch = torchvision.io.read_image(args.patch).float() / 255.0
    patch = patch.to(device).requires_grad_(False)
    print(f"Loaded patch from {args.patch}, shape: {patch.shape}")

    # Initialize patch transformer
    transformer = PatchTransformer(
        img_size=224,
        patch_size=args.patch_size,
        contrast=(0.9, 1.1),
        brightness=(-0.1, 0.1),
        angle=(-20, 20),
        noise_factor=0.05,
    )
    transformer = transformer.to(device)

    if args.defense:
        dwt_pool = dict(
            type="AvgPool2d",
            kernel_size=3,
            stride=1,
            padding=1,
        )
        dwt_preprocess = DWTPreprocessPlugin(
            max_level=5,
            threshold=0.17,
            pool=dwt_pool,
        )

    # Target class
    target_class = args.target_class
    print(f"Target class: {target_class}")

    # Evaluation loop
    total = 0
    correct = 0

    pbar = tqdm(dataloader, desc="Evaluating Patch")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Generate transformed patch
        patch_transformed, mask = transformer(patch, images.shape[0])

        # Apply patch to images
        adv_images = images * (1 - mask) + patch_transformed
        adv_images = torch.clamp(adv_images, 0, 1)

        if args.defense:
            adv_images = dwt_preprocess(adv_images)

        # Forward pass
        outputs = model(normalize(adv_images))

        # Calculate success rate
        _, predicted = outputs.max(1)
        correct += (predicted != target_class).sum().item()
        # correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        accuracy = 100.0 * correct / total
        pbar.set_postfix({"Accuracy": f"{accuracy:.2f}%"})

    # Calculate final statistics
    final_accuracy = 100.0 * correct / total
    print(f"Final Accuracy: {final_accuracy:.2f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Adversarial Patch Attack for Classification Models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/imagenet",
        help="Path to ImageNet dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/classifier_patches",
        help="Output path",
    )
    parser.add_argument("--weight", type=str, default="", help="Model weight")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patch-size", type=int, default=80, help="Patch size")
    parser.add_argument("--save-interval", type=int, default=5, help="Save interval")
    parser.add_argument("--target-class", type=int, default=0, help="Target class")
    parser.add_argument(
        "--defense", action="store_true", default=False, help="Apply defense"
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate only mode")
    parser.add_argument("--patch", type=str, help="Path to saved patch for evaluation")

    args = parser.parse_args()

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval:
        if not args.patch:
            parser.error("--patch is required in eval mode")
        evaluate_patch(args)
    else:
        train_patch(args)


if __name__ == "__main__":
    main()
