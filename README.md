# retail-shelf-pipeline

## Overview
End-to-end retail shelf analysis pipeline that runs, in order:
- Object detection (YOLO)
- Classification (EfficientNet)
- Segmentation (YOLOv8-seg + classification)
- OCR (YOLO + classification + EasyOCR)

## Architecture Diagrams
Part 1 architecture:
![architecture-1](https://github.com/SAMI7434/retail-shelf-pipeline/blob/main/Screenshot%202026-05-25%20193838.png)

Part 2 architecture:
![architecture-2](https://github.com/SAMI7434/retail-shelf-pipeline/blob/main/Screenshot%202026-05-25%20193928.png)


## Repository Structure
- detection.py: YOLO detection and training
- classification.py: EfficientNet classification and training
- segmentation.py: YOLOv8-seg + classification KPI plots
- ocr.py: YOLO + classification + OCR pipeline
- pipeline.py: orchestrates detection -> classification -> segmentation -> OCR
- best.pt: YOLO weights (put your trained weights here)
- best_efficientnet_b3.pth: classifier checkpoint
only 10 class =['Bingo', 'Britannia', 'Cheetos', 'doritos',
            'Kurkure', 'lays', 'Oreo', 'Parle', 'Pringles', 'Uncle Chipps']
## Setup
1) Create a Python environment (3.10+ recommended).
2) Install dependencies:

```bash
pip install ultralytics torch torchvision opencv-python matplotlib easyocr numpy
```

## Execution
### 1) Inference pipeline (all steps)
Update paths in pipeline.py if needed, then run:

```bash
python pipeline.py
```

### 2) Detection only
```bash
python detection.py
```

### 3) Classification only
```bash
python classification.py
```

### 4) Segmentation only
```bash
python segmentation.py
```

### 5) OCR only
```bash
python ocr.py
```

## Fine-tuning
### YOLO (detection)
Use detection.train_yolo in detection.py or call it from a small script. Example:

```python
from detection import train_yolo
train_yolo(data_yaml="/path/to/data.yaml")
```

### EfficientNet (classification)
Use classification.train_classifier in classification.py. Example:

```python
from classification import train_classifier
train_classifier(data_dir="/path/to/classification/dataset")
```
colab-https://colab.research.google.com/drive/1zrXWO5VSPrCI1HQiNJ6HkXrCoipDE6Lq#scrollTo=l6VrIXi5akVU

## Prediction Outputs (All Three Test Images)
Run inference on each of the three provided test images and save outputs.
Recommended: update pipeline.py to point to each image one at a time, or call
run_detection/run_classification/run_segmentation/run_ocr_pipeline directly.

Record and include outputs for all three images:
- model_performance.png (detection visualization)
- prediction.png (classification visualization)
- segmentation_result.png (segmentation KPI visualization)
- shelf_ocr_result.png (OCR overlay visualization)

## Visualized Prediction Plots
This repo generates and saves plots by default:
- model_performance.png (YOLO detections)
- prediction.png (classification confidence bars)
- segmentation_result.png (segmentation overlay + KPI bars)
- shelf_ocr_result.png (OCR overlays)

## Model/Tool Selection Rationale
- Ultralytics YOLO: fast and accurate object detection with easy fine-tuning.
- YOLOv8-seg: segmentation with masks for shelf space analysis.
- EfficientNet-B3: strong accuracy/size tradeoff for classification.
- EasyOCR: robust, open-source OCR for price/weight extraction.
- OpenCV + Matplotlib: reliable image IO and visualization.

#outputs
![image plt](https://github.com/SAMI7434/retail-shelf-pipeline/blob/c35d98bafb966dc839be06233de51afb31e9c4fd/output_images/Screenshot%202026-05-25%20032814.png)

## Assumptions, Limitations, and Tradeoffs
- Assumes input images are clear shelf images with visible products and price tags.
- OCR accuracy depends on resolution, lighting, and tag font size.
- Classification quality depends on dataset balance and label quality.
- Single-image pipeline; batch processing can be added if needed.
- Using YOLO and EfficientNet prioritizes speed and simplicity over exhaustive tuning.

## Submission Checklist
- Clean, modular code repository
- README with setup and execution steps
- Model/tool selection rationale
- Prediction outputs on all three test images
- Visualized prediction plots
- Summary of assumptions, limitations, and tradeoffs
