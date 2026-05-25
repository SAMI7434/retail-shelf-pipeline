"""End-to-end pipeline: detection -> classification -> segmentation -> OCR."""

from __future__ import annotations

from classification import run_classification
from detection import run_detection
from ocr import run_ocr_pipeline
from segmentation import run_segmentation


def main() -> None:
    # Detection
    det_weights = "/content/drive/MyDrive/retail-shelf-analyzer/runs/product_detector/weights/best.pt"
    det_image = "/img_2.jpg"
    run_detection(det_weights, det_image)

    # Classification
    clf_checkpoint = "best_efficientnet_b3.pth"
    clf_image = "/img_2.jpg"
    run_classification(clf_checkpoint, clf_image)

    # Segmentation
    seg_weights = "yolov8n-seg.pt"
    seg_image = "/img_2.jpg"
    run_segmentation(seg_weights, clf_checkpoint, seg_image)

    # OCR
    ocr_weights = det_weights
    ocr_image = "/img_2.jpg"
    run_ocr_pipeline(ocr_weights, clf_checkpoint, ocr_image)


if __name__ == "__main__":
    main()
