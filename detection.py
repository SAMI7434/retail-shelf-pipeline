"""Detection pipeline using YOLO."""

from collections import Counter

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def run_detection(
    weights_path: str,
    image_path: str,
    conf: float = 0.05,
    iou: float = 0.4,
    imgsz: int = 1280,
) -> None:
    """Run YOLO detection and visualize results."""
    model = YOLO(weights_path)
    print("Model loaded")
    print(f"Classes: {model.names}")

    results = model(
        image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        agnostic_nms=True,
        verbose=False,
    )[0]

    print(f"Detected {len(results.boxes)} products")

    annotated = results.plot(
        line_width=2,
        font_size=10,
    )[:, :, ::-1]

    plt.figure(figsize=(18, 12))
    plt.imshow(annotated)
    plt.title(
        f"Model: product_detector | Detected: {len(results.boxes)} objects",
        fontsize=13,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("model_performance.png", dpi=150, bbox_inches="tight")
    plt.show()

    counts = Counter(model.names[int(b.cls)] for b in results.boxes)
    print("Detections per class:")
    for cls, cnt in counts.most_common():
        print(f"  {cls:<20} x {cnt}")


def train_yolo(
    data_yaml: str,
    base_weights: str = "yolo26n.pt",
    epochs: int = 60,
    imgsz: int = 1280,
    batch: int = 8,
    project: str = "/content/drive/MyDrive/retail-shelf-analyzer/runs",
    name: str = "product_detector",
    device: int | str = 0,
    patience: int = 15,
) -> None:
    """Train a YOLO model and save weights to the project folder."""
    model = YOLO(base_weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=project,
        device=device,
        patience=patience,
        save=True,
        workers=2,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
    )

    print("\n✅ Training complete!")
    print("Best weights → runs/product_detector/weights/best.pt")


if __name__ == "__main__":
    WEIGHTS = "/content/drive/MyDrive/retail-shelf-analyzer/runs/product_detector/weights/best.pt"
    IMAGE_PATH = "/img_2.jpg"
    run_detection(WEIGHTS, IMAGE_PATH)
