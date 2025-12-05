import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

# 여기 경로는 실제 패키지 구조에 따라 조정 필요
from src.dataset.pill_dataset import PillDataset
from src.models.faster_rcnn import get_faster_rcnn_model


def collate_fn(batch):
    """torchvision detection 모델용 collate_fn."""
    return tuple(zip(*batch))


def train_one_epoch(
    model,
    optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict: Dict[str, torch.Tensor] = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if step % 10 == 0:
            print(
                f"[Epoch {epoch}] Step {step}/{len(data_loader)} "
                f"loss: {losses.item():.4f}"
            )

    epoch_loss = running_loss / max(1, len(data_loader))
    return epoch_loss


def main():
    # ====== TODO: 실제 데이터 경로 & 라벨 로드 ======
    # 예시:
    #   img_dir = "data/raw/train_images"
    #   ann_path = "data/annotations/train.csv"
    #   annotations = pd.read_csv(ann_path)
    #   dataset = PillDataset(img_dir, annotations, transforms=None)
    #
    # 지금은 템플릿이므로, 실제 데이터셋 구조를 확인한 뒤
    # 팀에서 함께 수정하는 단계에서 채워 넣으면 됨.
    import pandas as pd  # TODO: 필요 시 requirements에 반영

    img_dir = "data/raw"  # TODO: 실제 폴더명으로 수정
    ann_path = "data/annotations/train.csv"  # TODO

    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"라벨 파일을 찾을 수 없습니다: {ann_path}\n"
            "→ competition에서 받은 train.csv 위치를 확인하고 수정해주세요."
        )

    annotations = pd.read_csv(ann_path)

    dataset = PillDataset(
        img_dir=img_dir,
        annotations=annotations,
        transforms=None,  # TODO: 나중에 augmentation 추가
    )

    # 간단히 train/val 나누기 (예시)
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== 모델 생성 ======
    num_classes = 11  # 예시: 배경 + 10개 알약 → 나중에 실제 클래스 수로 수정
    model = get_faster_rcnn_model(num_classes=num_classes)
    model.to(device)

    # ====== Optimizer ======
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    num_epochs = 2  # 처음엔 작게 (동작 확인용)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        # TODO: val_loader로 검증 로직 추가 (mAP, loss 등)

    # TODO: 모델 저장 예시
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", "faster_rcnn_baseline.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()

