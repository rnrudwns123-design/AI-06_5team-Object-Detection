# AI-06_5team-Object-Detection
Team-based oral medication object detection project (Kaggle Competition)
# 🧪 HealthEat Pill Detection Project

경구약제 이미지에서 **알약의 종류(클래스)와 위치(바운딩 박스)** 를 예측하는  
객체 탐지(Object Detection) 팀 프로젝트입니다.

Kaggle Private Competition에 제출하여 성능을 확인하고,  
프로젝트 과정을 협업 일지/보고서/발표로 정리하는 것이 목표입니다.

---

## 📁 Project Structure

```text
AI-06_5team-Object-Detection/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # 원본 이미지/라벨 (수정 X)
│   ├── processed/          # 필요하면 전처리된 데이터
│   ├── annotations/        # 라벨 CSV/JSON/XML 등
│   └── sample_submission/  # Kaggle 제출 예시
│
├── src/
│   ├── dataset/
│   │   └── pill_dataset.py   # Dataset 클래스
│   ├── models/
│   │   └── faster_rcnn.py    # Faster R-CNN 모델 정의
│   ├── training/
│   │   └── train.py          # 학습 스크립트 (baseline)
│   ├── inference/
│   │   └── (TODO)            # 추론/시각화/제출 생성 코드
│   └── config/
│       └── (TODO)            # 설정 파일 (yaml 등) 필요 시
│
├── notebooks/
│   ├── EDA.ipynb             # 데이터 탐색용 노트북
│   └── experiments.ipynb     # 실험 기록용 노트북
│
└── docs/
    ├── report/               # 최종 보고서, md/pdf
    ├── slides/               # 발표 자료
    └── logs/                 # 실험 로그/결과 정리

Team & Roles

(임시로 비워두고 팀 회의 때 채워넣기)

PM / Scrum Master:

Data Engineer:

Model Architect:

Experimentation Lead:

역할은 유연하게, 여러 명이 같이 맡아도 괜찮습니다.

🔧 Environment Setup
# (선택) 가상환경 생성
python -m venv venv
.\venv\Scripts\activate  # Windows PowerShell 기준

# 패키지 설치
pip install -r requirements.txt

🚀How to Use (Baseline Flow)

1. 데이터 다운로드 & 배치

제공된 경구약제 이미지/라벨을 data/ 아래에 배치

예:

이미지: data/raw/

라벨: data/annotations/train.csv (형식은 나중에 팀에서 맞추기)

2. Dataset 코드 수정 (src/dataset/pill_dataset.py)

실제 라벨 CSV/JSON 형식에 맞게

이미지 경로

바운딩 박스

클래스 라벨
를 읽어서 PyTorch Dataset 형태로 리턴하도록 수정

3. 모델 학습 (src/training/train.py)
python -m src.training.train

초기에 적은 epoch로 baseline 성능 확인

이후 하이퍼파라미터/모델 구조 수정하면서 개선

🧪 Baseline Model

Backbone: Faster R-CNN ResNet50 FPN (pretrained on COCO) 

Loss: 기본 Faster R-CNN loss

Metric: mAP / Kaggle Leaderboard Score

📓 Collaboration Rules (Quick)

main 브랜치: 항상 “동작하는 상태” 유지

각자 기능 작업은 브랜치 파서 진행 (예: feature/dataset, feature/yolo)

커밋 메시지: 짧고 의미 있게 (예: Add basic PillDataset, Fix bbox normalization)

📝 협업 일지 (각자)

각 팀원은 개인 Notion/Docs/Markdown으로 협업 일지를 작성:

오늘 할 일 → 오늘 한 일 → 어려웠던 점 → 팀 기여 포인트

스프린트 종료 후, 이를 바탕으로

발표

포트폴리오

회고 정리
에 활용합니다. 

## How to Start

1) Clone the repository  
2) Install dependencies  
3) Run simple training: python src/training/train.py  
4) Notebook-based EDA: notebooks/EDA.ipynb  
5) Submission 생성: python src/inference/make_submission.py
