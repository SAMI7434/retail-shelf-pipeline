"""Segmentation pipeline using YOLOv8-seg and classification."""

from collections import Counter

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


def run_segmentation(
    yolo_seg_weights: str,
    classifier_checkpoint: str,
    image_path: str,
    device: str | None = None,
) -> None:
    """Run segmentation + classification and visualize results."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(classifier_checkpoint, map_location=device)
    classes = ckpt["classes"]
    num_classes = ckpt["num_classes"]

    classifier = models.efficientnet_b3(weights=None)
    classifier.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(classifier.classifier[1].in_features, num_classes),
    )
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    total_pixels = h * w

    seg_model = YOLO(yolo_seg_weights)
    results = seg_model(
        image_path,
        conf=0.05,
        iou=0.4,
        imgsz=720,
        agnostic_nms=True,
        verbose=False,
    )[0]

    print(f"Detected {len(results.boxes)} products")
    has_masks = results.masks is not None
    print(f"Masks available: {has_masks}")

    cmap = plt.cm.get_cmap("tab10", num_classes)
    cls_color = {
        c: (np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i, c in enumerate(classes)
    }

    detections: list[dict] = []
    overlay = image_rgb.copy().astype(np.float32)
    shelf_width = w

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
        x1c = max(0, x1 - px)
        y1c = max(0, y1 - py)
        x2c = min(w, x2 + px)
        y2c = min(h, y2 + py)

        crop = Image.fromarray(image_rgb[y1c:y2c, x1c:x2c])
        tensor = transform(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(classifier(tensor), dim=1)[0]
        conf, idx = probs.max(0)
        label = classes[idx]
        color = cls_color[label]

        mask_pixels = 0
        shelf_pct = 0.0

        if has_masks and results.masks.data is not None:
            mask_tensor = results.masks.data[i]
            mask_np = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST).astype(
                bool
            )

            mask_pixels = int(mask_resized.sum())
            shelf_pct = 100.0 * mask_pixels / total_pixels

            overlay[mask_resized] = overlay[mask_resized] * 0.4 + color * 0.6
        else:
            mask_pixels = (x2 - x1) * (y2 - y1)
            shelf_pct = 100.0 * mask_pixels / total_pixels

        facing_width = x2 - x1
        facing_pct = 100.0 * facing_width / shelf_width

        detections.append(
            {
                "label": label,
                "conf": conf.item(),
                "bbox": (x1, y1, x2, y2),
                "mask_pixels": mask_pixels,
                "shelf_pct": shelf_pct,
                "facing_width": facing_width,
                "facing_pct": facing_pct,
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
            }
        )

    fig, axes = plt.subplots(2, 1, figsize=(22, 20))

    ax1 = axes[0]
    ax1.imshow(overlay.astype(np.uint8))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color_f = tuple(cls_color[det["label"]] / 255.0)

        ax1.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.5,
                edgecolor=color_f,
                facecolor="none",
            )
        )
        ax1.text(
            x1,
            y1 - 4,
            f"{det['label']} {det['conf']*100:.0f}%",
            color="white",
            fontsize=6,
            fontweight="bold",
            bbox=dict(facecolor=color_f, alpha=0.9, pad=1.5, edgecolor="none"),
        )
        ax1.text(
            x1,
            y2 + 3,
            f"{det['shelf_pct']:.1f}%",
            color="white",
            fontsize=6,
            bbox=dict(facecolor="#2c3e50", alpha=0.8, pad=1, edgecolor="none"),
        )

    ax1.set_title(
        f"Segmentation + Classification | {len(detections)} products",
        fontsize=13,
        fontweight="bold",
    )
    ax1.axis("off")

    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    brand_space: dict[str, float] = {}
    brand_facings: dict[str, int] = {}
    brand_colors: dict[str, tuple] = {}

    for det in detections:
        lbl = det["label"]
        brand_space[lbl] = brand_space.get(lbl, 0) + det["shelf_pct"]
        brand_facings[lbl] = brand_facings.get(lbl, 0) + 1
        brand_colors[lbl] = tuple(cls_color[lbl] / 255.0)

    sorted_brands = sorted(brand_space.items(), key=lambda x: -x[1])
    brand_names = [b[0] for b in sorted_brands]
    space_vals = [b[1] for b in sorted_brands]
    facing_vals = [brand_facings[b] for b in brand_names]
    bar_colors = [brand_colors[b] for b in brand_names]

    x = np.arange(len(brand_names))
    width = 0.4

    bars1 = ax2.bar(
        x - width / 2, space_vals, width, color=bar_colors, alpha=0.9, label="Shelf Space %"
    )
    bars2 = ax2.bar(
        x + width / 2,
        facing_vals,
        width,
        color=bar_colors,
        alpha=0.5,
        label="Facings count",
        edgecolor="white",
        linewidth=0.5,
    )

    for bar in bars1:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{bar.get_height():.1f}%",
            ha="center",
            color="white",
            fontsize=8,
        )

    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(int(bar.get_height())),
            ha="center",
            color="white",
            fontsize=8,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(brand_names, color="white", fontsize=10, rotation=20, ha="right")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444")
    ax2.yaxis.label.set_color("white")
    ax2.set_ylabel("Value", color="white")
    ax2.set_title(
        "Shelf Space % vs Facings per Brand",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(facecolor="#1a1a2e", edgecolor="white", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig("segmentation_result.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.show()

    print("SHELF KPI REPORT")
    print("=" * 55)
    print(f"  Total products detected : {len(detections)}")
    print(f"  Image size              : {w} x {h} px")
    print(f"\n{'Brand':<15} {'Facings':>8} {'Shelf Space':>12} {'Avg Width':>10}")
    print("-" * 55)

    for brand in brand_names:
        dets = [d for d in detections if d["label"] == brand]
        avg_w = np.mean([d["facing_width"] for d in dets])
        print(
            f"  {brand:<13} {brand_facings[brand]:>8} "
            f"{brand_space[brand]:>10.1f}%  {avg_w:>8.0f}px"
        )

    print("\nGap Detection (empty shelf spaces):")
    sorted_by_x = sorted(detections, key=lambda d: d["bbox"][0])
    gaps = []
    for i in range(len(sorted_by_x) - 1):
        curr_x2 = sorted_by_x[i]["bbox"][2]
        next_x1 = sorted_by_x[i + 1]["bbox"][0]
        gap = next_x1 - curr_x2
        if gap > 50:
            gaps.append(
                {
                    "x": curr_x2,
                    "width": gap,
                    "between": f"{sorted_by_x[i]['label']} -> {sorted_by_x[i+1]['label']}",
                }
            )

    if gaps:
        for g in gaps:
            print(f"  Gap of {g['width']}px at x={g['x']} ({g['between']})")
    else:
        print("  No significant gaps detected")


if __name__ == "__main__":
    YOLO_SEG = "yolov8n-seg.pt"
    CLASSIFIER_PT = "best_efficientnet_b3.pth"
    IMAGE_PATH = "/img_2.jpg"
    run_segmentation(YOLO_SEG, CLASSIFIER_PT, IMAGE_PATH)
