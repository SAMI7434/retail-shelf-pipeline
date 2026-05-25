"""Classification pipeline using EfficientNet."""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def run_classification(
    checkpoint_path: str,
    image_path: str,
    device: str | None = None,
) -> None:
    """Run EfficientNet classification and visualize results."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]
    num_classes = ckpt["num_classes"]

    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    sorted_results = sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])
    top_class, top_prob = sorted_results[0]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    ax_img.imshow(np.array(img))
    ax_img.axis("off")
    ax_img.text(
        0.5,
        0.04,
        f"{top_class}  {top_prob*100:.1f}%",
        transform=ax_img.transAxes,
        fontsize=13,
        fontweight="bold",
        color="white",
        ha="center",
        bbox=dict(facecolor="#2ecc71", alpha=0.9, pad=4, edgecolor="none"),
    )

    labels = [r[0] for r in sorted_results]
    values = [r[1] * 100 for r in sorted_results]
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(labels))]

    bars = ax_bar.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax_bar.set_xlim(0, 115)
    ax_bar.set_facecolor("#16213e")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444")
    ax_bar.xaxis.label.set_color("white")
    ax_bar.set_xlabel("Confidence %", color="white")
    ax_bar.set_title("All Classes", color="white", fontsize=12, fontweight="bold")
    ax_bar.yaxis.set_tick_params(labelcolor="white", labelsize=10)

    plt.suptitle(
        f"Prediction: {top_class}  ({top_prob*100:.1f}%)",
        color="white",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("prediction.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()

    print(f"Prediction: {top_class}  ({top_prob*100:.1f}%)")
    print("All classes:")
    for cls, prob in sorted_results:
        print(f"  {cls:<15} {prob*100:5.1f}%")


def train_classifier(
    data_dir: str,
    model_size: str = "b3",
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-4,
    early_stop_patience: int = 10,
    image_size: int = 300,
    device: str | None = None,
) -> None:
    """Train EfficientNet classifier and save best weights."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform
    )

    num_classes = len(train_dataset.classes)
    assert train_dataset.classes == val_dataset.classes, "Class mismatch!"

    train_fnames = {os.path.basename(p) for p, _ in train_dataset.samples}
    val_fnames = {os.path.basename(p) for p, _ in val_dataset.samples}
    overlap = train_fnames & val_fnames
    if overlap:
        print(f"⚠️  {len(overlap)} files in BOTH train and val — fix this!")
    else:
        print("✅ No overlap between train and val")

    print(f"Classes ({num_classes}): {train_dataset.classes}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    with open("class_names.json", "w") as f:
        json.dump(
            {
                "classes": train_dataset.classes,
                "num_classes": num_classes,
                "model_size": model_size,
                "image_size": image_size,
            },
            f,
            indent=2,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nTotal: {total:,} | Trainable: {trainable:,} "
        f"({100*trainable/total:.1f}%)"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
    )

    def train_epoch(model, loader):
        model.train()
        total_loss, correct, total_count = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += model(images).detach().max(1)[1].eq(labels).sum().item()
            total_count += labels.size(0)
        return total_loss / len(loader), 100.0 * correct / total_count

    def validate(model, loader):
        model.eval()
        total_loss, correct, total_count = 0.0, 0, 0
        class_correct, class_total = {}, {}
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()
                preds = outputs.max(1)[1]
                correct += preds.eq(labels).sum().item()
                total_count += labels.size(0)
                for l, p in zip(labels, preds):
                    li = l.item()
                    class_correct[li] = class_correct.get(li, 0) + (li == p.item())
                    class_total[li] = class_total.get(li, 0) + 1
        per_class = {
            train_dataset.classes[i]: 100.0 * class_correct[i] / class_total[i]
            for i in class_total
        }
        return total_loss / len(loader), 100.0 * correct / total_count, per_class

    best_acc, no_improve = 0.0, 0

    print("\n" + "=" * 60)
    print(f"TRAINING  {num_classes} classes: {train_dataset.classes}")
    print("=" * 60)

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader)
        val_loss, val_acc, per_class = validate(model, val_loader)
        scheduler.step(val_acc)

        print(f"\nEpoch [{epoch+1}/{epochs}] | {time.time()-start:.1f}s")
        print(f"  Train: {train_loss:.4f} loss | {train_acc:.1f}% acc")
        print(f"  Val:   {val_loss:.4f} loss | {val_acc:.1f}% acc")
        print(f"  Per-class: { {k: f'{v:.0f}%' for k, v in per_class.items()} }")

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": train_dataset.classes,
                    "num_classes": num_classes,
                },
                "best_efficientnet_b3.pth",
            )
            print(f"  >>> ✅ Saved best model (Val Acc: {val_acc:.1f}%)")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{early_stop_patience})")
            if no_improve >= early_stop_patience:
                print(f"\n⏹  Early stopping at epoch {epoch+1}")
                break

    print(f"\n✅ DONE | Best Val Acc: {best_acc:.1f}%")
    print("Saved: best_efficientnet_b3.pth")


if __name__ == "__main__":
    CHECKPOINT = "best_efficientnet_b3.pth"
    IMAGE_PATH = "/img_2.jpg"
    run_classification(CHECKPOINT, IMAGE_PATH)
