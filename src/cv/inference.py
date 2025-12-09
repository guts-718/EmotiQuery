# src/cv/inference.py

import argparse
from pathlib import Path
import csv
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, f1_score

from .model_baseline import EmotionCNN
from .model_resnet import EmotionResNet18


def get_project_paths() -> Tuple[Path, Path, Path]:
    """
    Compute useful project paths based on this file's location.

    Returns:
        project_root, data_dir, outputs_dir
    """
    # .../project_root/src/cv/inference.py -> parents[2] = project_root
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "src" / "data"
    outputs_dir = project_root / "outputs"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    return project_root, data_dir, outputs_dir


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom ImageFolder that also returns the image file path.

    __getitem__ returns: (image_tensor, label, path)
    """

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Create model based on name and move it to device.
    For inference we don't need pretrained=True because we load a checkpoint.
    """
    if model_name == "baseline":
        model = EmotionCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        model = EmotionResNet18(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


def get_test_loader(test_dir: Path, batch_size: int, num_workers: int = 2):
    """
    Create a DataLoader for the test dataset, returning (image, label, path).
    """
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_dataset = ImageFolderWithPaths(str(test_dir), transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return test_loader, test_dataset.classes


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on test set and export predictions CSV.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["baseline", "resnet18"],
        help="Which model architecture to use for loading checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional custom path for predictions CSV (relative to project root).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root, data_dir, outputs_dir = get_project_paths()
    test_dir = data_dir / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Expected test data at {test_dir}, but not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Class names from checkpoint if available, else from dataset later
    ckpt_class_names: List[str] = checkpoint.get("class_names", [])

    # Data
    test_loader, dataset_class_names = get_test_loader(
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=2 if device.type == "cuda" else 0,
    )

    if ckpt_class_names:
        class_names = ckpt_class_names
        print(f"Using class names from checkpoint: {class_names}")
    else:
        class_names = dataset_class_names
        print(f"Using class names from dataset: {class_names}")

    num_classes = len(class_names)

    # Model
    model = build_model(args.model, num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_true: List[int] = []
    all_pred: List[int] = []
    all_conf: List[float] = []
    all_paths: List[str] = []

    softmax = nn.Softmax(dim=1)

    print("Running inference on test set...")

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = softmax(outputs)

            confidences, preds = torch.max(probs, dim=1)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_conf.extend(confidences.cpu().tolist())
            all_paths.extend(paths)

    # Metrics
    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro")

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Total samples: {len(all_true)}")

    # Determine output CSV path
    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        if not output_csv_path.is_absolute():
            output_csv_path = project_root / output_csv_path
    else:
        output_csv_path = outputs_dir / f"predictions_{args.model}_test.csv"

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    print(f"Writing predictions to: {output_csv_path}")
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "true_label",
            "true_index",
            "predicted_label",
            "predicted_index",
            "confidence",
        ])

        for path, t_idx, p_idx, conf in zip(all_paths, all_true, all_pred, all_conf):
            true_label = class_names[t_idx] if 0 <= t_idx < len(class_names) else str(t_idx)
            pred_label = class_names[p_idx] if 0 <= p_idx < len(class_names) else str(p_idx)

            # Store paths relative to project root if possible
            try:
                rel_path = str(Path(path).resolve().relative_to(project_root))
            except ValueError:
                rel_path = path

            writer.writerow([
                rel_path,
                true_label,
                t_idx,
                pred_label,
                p_idx,
                f"{conf:.6f}",
            ])

    print("Done.")


if __name__ == "__main__":
    main()
