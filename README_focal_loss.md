# Focal Loss 실험 기록 (YOLOv8s)

## 📌 실험 목적

기존 YOLOv8s의 BCE 기반 분류 손실 대신  
Focal Loss를 적용하여 클래스 불균형과 hard sample에 대한 탐지 성능 개선 여부를 평가함.

---

## 📌 수정된 소스 코드

### 1. custom_ultralytics/loss_focal.py

Ultralytics의 `ultralytics/utils/loss.py`를 기반으로  
`SimpleFocalLoss`를 추가하고 `v8DetectionLoss`가 이를 사용하도록 수정한 코드 사본.

> 로컬 Python env에서는 이 파일의 내용을 원본 `loss.py`에 반영하여 학습을 수행함.

---

## 📌 학습 커맨드

```bash
yolo detect train \
  model=yolov8s.pt \
  data=notebooks/data/yolo_dataset/data.yaml \
  epochs=30 \
  imgsz=512 \
  batch=8 \
  name=train3
