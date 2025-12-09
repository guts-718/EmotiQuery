# src/cv/train.py

import os
import argparse
import csv
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

from .dataset import get_data_loaders
from .model_baseline import EmotionCNN
from .model_resnet import EmotionResNet18


def get_project_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Compute useful project paths based on this file's location.

    Returns:
        project_root, data_dir, models_dir, logs_dir
    """
    # .../project_root/src/cv/train.py -> parents[2] = project_root
    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "src" / "data"
    models_dir = project_root / "models"
    logs_dir = project_root / "logs"

    # Ensure dirs exist
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return project_root, data_dir, models_dir, logs_dir


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """
    Create model based on name and move it to device.
    """
    if model_name == "baseline":
        model = EmotionCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        model = EmotionResNet18(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch and return average loss.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    return epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate the model on a dataloader.

    Returns:
        avg_loss, accuracy, macro_f1
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, macro_f1


def init_csv_logger(log_path: Path):
    """
    Create CSV file with header if it does not exist.
    """
    if not log_path.exists():
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1"])


def append_metrics(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float,
    val_f1: float,
):
    """
    Append metrics for a single epoch to CSV.
    """
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, val_acc, val_f1])


def parse_args():
    parser = argparse.ArgumentParser(description="Train facial emotion recognition model.")

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "resnet18"],
        help="Which model to train: 'baseline' or 'resnet18'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (keep small on CPU)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root, data_dir, models_dir, logs_dir = get_project_paths()

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected data at {train_dir} and {test_dir}, but not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    train_loader, test_loader, class_names = get_data_loaders(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        batch_size=args.batch_size,
        img_size=(48, 48),
        num_workers=2 if device.type == "cuda" else 0,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # 2. Model
    model = build_model(args.model, num_classes=num_classes, device=device)
    print(f"Training model: {args.model}")

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. Logging setup
    log_path = logs_dir / f"training_log_{args.model}.csv"
    init_csv_logger(log_path)


    # TensorBoard Writer (NEW)
    tb_log_dir = logs_dir / f"tensorboard_{args.model}"
    writer = SummaryWriter(log_dir=str(tb_log_dir))

    best_val_f1 = 0.0
    best_model_path = models_dir / f"{args.model}_best.pt"

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 30)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, test_loader, criterion, device)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

      
        # Log to CSV + TensorBoard
        append_metrics(log_path, epoch, train_loss, val_loss, val_acc, val_f1)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("F1/Validation", val_f1, epoch)


        # Save best model based on macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "class_names": class_names,
                    "model_name": args.model,
                },
                best_model_path,
            )
            print(f"🔥 New best model saved to: {best_model_path}")
    
    writer.close()

    print("\nTraining complete.")
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    print(f"Best model stored at: {best_model_path}")


if __name__ == "__main__":
    main()
