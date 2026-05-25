"""OCR pipeline: YOLO detection + classification + EasyOCR."""

import re
from collections import Counter

import cv2
import easyocr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


def parse_price(texts: list[str]) -> tuple[str | None, str | None]:
    """Extract price and weight from OCR results."""
    price = None
    weight = None
    full = " ".join(texts).upper()

    price_match = re.search(r"[\$RS\.]*\s*(\d{1,4}(?:\.\d{1,2})?)", full)
    if price_match:
        price = f"{price_match.group(1)}"

    weight_match = re.search(r"(\d+(?:\.\d+)?)\s*[Gg]", full)
    if weight_match:
        weight = f"{weight_match.group(1)}g"

    return price, weight


def run_ocr_pipeline(
    yolo_weights: str,
    classifier_checkpoint: str,
    image_path: str,
    device: str | None = None,
) -> None:
    """Run detection + classification + OCR for price tags."""
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

    yolo = YOLO(yolo_weights)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    results = yolo(
        image_path,
        conf=0.25,
        iou=0.4,
        imgsz=1280,
        agnostic_nms=True,
        verbose=False,
    )[0]
    print(f"YOLO detected {len(results.boxes)} products")

    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    detections: list[dict] = []
    for box in results.boxes:
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
        cls_conf, idx = probs.max(0)

        price_y1 = y2
        price_y2 = min(h, y2 + 80)
        price_x1 = max(0, x1 - 10)
        price_x2 = min(w, x2 + 10)

        price_region = image_rgb[price_y1:price_y2, price_x1:price_x2]

        price, weight = None, None
        if price_region.size > 0 and price_region.shape[0] > 10:
            gray = cv2.cvtColor(price_region, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            ocr_results = reader.readtext(upscaled, detail=0, paragraph=True)
            price, weight = parse_price(ocr_results)

        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "label": classes[idx],
                "conf": cls_conf.item(),
                "price": price,
                "weight": weight,
            }
        )

    cmap = plt.cm.get_cmap("tab10", num_classes)
    cls_color = {c: cmap(i) for i, c in enumerate(classes)}

    fig, ax = plt.subplots(figsize=(22, 14))
    ax.imshow(image_rgb)

    counts = Counter()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        color = cls_color[label]
        counts[label] += 1

        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )

        ax.text(
            x1,
            y1 - 5,
            f"{label} {det['conf']*100:.0f}%",
            color="white",
            fontsize=7,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.9, pad=2, edgecolor="none"),
        )

        if det["price"] or det["weight"]:
            price_txt = " | ".join(filter(None, [det["price"], det["weight"]]))
            ax.text(
                x1,
                y2 + 5,
                price_txt,
                color="white",
                fontsize=7,
                fontweight="bold",
                bbox=dict(facecolor="#e74c3c", alpha=0.9, pad=2, edgecolor="none"),
            )

    ax.set_title(
        f"Shelf Analysis | {len(detections)} products | Prices via OCR",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("shelf_ocr_result.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Product Summary:")
    print(f"{'Product':<15} {'Confidence':<12} {'Price':<8} {'Weight':<8}")
    print("-" * 45)
    for det in detections:
        price = det["price"] or "N/A"
        weight = det["weight"] or "N/A"
        print(
            f"  {det['label']:<13} {det['conf']*100:>6.1f}%     "
            f"{price:<8} {weight:<8}"
        )

    print("Count per brand:")
    for product, count in counts.most_common():
        print(f"  {product:<15} x {count}")


if __name__ == "__main__":
    YOLO_WEIGHTS = "/content/drive/MyDrive/retail-shelf-analyzer/runs/product_detector/weights/best.pt"
    CLASSIFIER_PT = "best_efficientnet_b3.pth"
    IMAGE_PATH = "/img_2.jpg"
    run_ocr_pipeline(YOLO_WEIGHTS, CLASSIFIER_PT, IMAGE_PATH)
