
import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    resnet18,
    ResNet18_Weights,
)


NUM_CLASSES = 4

CLASS_NAMES = [
    "pawara",
    "uchitha",
    "dhamith",
    "unknown",
]


# ==========================================
# MODEL A: CUSTOM CNN
# ==========================================

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ==========================================
# MODEL B: MOBILENETV3-SMALL
# ==========================================

def build_mobilenet(
    num_classes=4,
    dropout=0.3,
    pretrained=True,
    freeze_backbone=False,
):
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained else None
    )

    model = mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features

    model.classifier[2] = nn.Dropout(p=dropout)
    model.classifier[-1] = nn.Linear(
        in_features, num_classes
    )

    return model


# ==========================================
# MODEL C: RESNET18
# ==========================================

def build_resnet(
    num_classes=4,
    dropout=0.3,
    pretrained=True,
    freeze_backbone=False,
):
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained else None
    )

    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


# ==========================================
# SHARED MODEL FACTORY
# ==========================================

def create_model(
    model_name,
    num_classes=NUM_CLASSES,
    dropout=0.3,
    pretrained=True,
    freeze_backbone=False,
):
    name = model_name.lower()

    if name == "custom_cnn":
        return CustomCNN(
            num_classes=num_classes,
            dropout=dropout,
        )

    if name == "mobilenet_v3":
        return build_mobilenet(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name == "resnet18":
        return build_resnet(
            num_classes=num_classes,
            dropout=dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )
