# AI-06_5team-Object-Detection
Team-based oral medication object detection project (Kaggle Competition)
# HealthEat Pill Detection Project

팀 기반 경구약제 객체 탐지(Object Detection) 프로젝트입니다.  
이미지 속 알약의 **클래스(이름)** 및 **바운딩 박스 좌표**를 예측하는 모델을 개발하고  
Kaggle Private Competition에 제출하는 것이 목표입니다.

---

## 📁 Repository Structure

- `data/`  
  - 원본 이미지 및 라벨
- `src/`  
  - 모델, 데이터셋, 학습/추론 코드
- `notebooks/`  
  - EDA, 실험용 Jupyter Notebook
- `docs/`  
  - 보고서, 발표자료, 실험 로그

---

## 🚀 Getting Started

## 프로젝트 개요

- 경구약제 이미지(Object Detection) 프로젝트
- 이미지 속 알약의 위치(바운딩 박스)와 이름(클래스)을 예측하는 모델을 개발
- Kaggle Private Competition에 제출하여 성능 비교

## 코드 구조(초기 버전)

- `data/` : 데이터 관련 폴더 (원본, 전처리, 라벨 등)
- `src/` : 실제 파이썬 코드
  - `dataset/` : 데이터셋 클래스
  - `models/` : 모델 정의 (Faster R-CNN 등)
  - `training/` : 학습 스크립트
  - `inference/` : 추론 및 제출 파일 생성
- `notebooks/` : EDA, 실험용 노트북
- `docs/` : 보고서, 발표자료, 로그

## 팀원들을 위한 간단한 규칙(초안)

- `main` 브랜치는 항상 "돌아가는 코드"만 유지
- 새로운 기능을 만들 땐 브랜치 사용 권장
  - 예: `feature/dataset`, `feature/baseline-model`
- 커밋 메시지는 짧게 영어/한국어 아무거나 OK
  - 예: `Add baseline FasterRCNN model`, `데이터셋 클래스 기본 구조 추가`
### 1. Install dependencies