# train_model.py
# 목적: anime vs Real 이미지 분류 모델 학습

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ----- 기본 설정 -----
BATCH_SIZE = 32
EPOCHS = 20  # 더 오래 학습
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 데이터 경로 -----
train_dir = "dataset/train"
val_dir = "dataset/val"

# ----- 클래스 개수 -----
NUM_CLASSES = len(os.listdir(train_dir))

# ----- 데이터 전처리 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # MobileNetV2용
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 클래스 라벨 확인
print("클래스 라벨 순서 (ImageFolder 기준):", train_dataset.class_to_idx)

# ----- 모델 불러오기 (사전학습 MobileNetV2) -----
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- 학습 및 검증 -----
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ----- 검증 -----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"[{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f} | 검증 정확도: {val_accuracy:.2f}%")

# ----- 모델 저장 -----
torch.save(model.state_dict(), "model.pth")
print("모델 저장 완료: model.pth")

print(train_dataset.class_to_idx)
