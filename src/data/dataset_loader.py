
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
TENSOR_DIR = ROOT / "data/tensors"

CLASS_NAMES = [
    "pawara",
    "uchitha",
    "dhamith",
    "unknown"
]

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CUSTOM_MEAN = (0.5, 0.5, 0.5)
CUSTOM_STD = (0.5, 0.5, 0.5)


# ==========================================
# MODEL-SPECIFIC TRANSFORMS
# ==========================================

def get_transforms(model_name, training=False):

    if model_name in ("mobilenet_v3", "resnet18"):
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

    elif model_name == "custom_cnn":
        mean = CUSTOM_MEAN
        std = CUSTOM_STD

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    operations = [
        transforms.ToPILImage()
    ]

    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10
            )
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transforms.Compose(operations)


# ==========================================
# PYTORCH TENSOR DATASET
# ==========================================

class FaceOpsDataset(Dataset):

    def __init__(
        self,
        split,
        model_name,
        augment=False
    ):

        if split not in ("train", "val", "test"):
            raise ValueError("Invalid dataset split")

        if augment and split != "train":
            raise ValueError(
                "Augmentation is allowed only for training"
            )

        dataset_path = TENSOR_DIR / f"{split}.pt"

        data = torch.load(
            dataset_path,
            map_location="cpu",
            weights_only=True
        )

        self.images = data["images"]
        self.labels = data["labels"].long()

        if self.images.ndim != 4:
            raise ValueError("Expected 4D image tensor")

        if self.images.shape[1:] != (3, 224, 224):
            raise ValueError("Invalid image dimensions")

        if self.images.dtype != torch.uint8:
            raise ValueError("Expected uint8 images")

        if len(self.images) != len(self.labels):
            raise ValueError("Image/label count mismatch")

        if list(data["class_names"]) != CLASS_NAMES:
            raise ValueError("Class mapping mismatch")

        if not torch.all(
            (self.labels >= 0) & (self.labels < 4)
        ):
            raise ValueError("Invalid class labels")

        self.split = split
        self.model_name = model_name

        self.transform = get_transforms(
            model_name,
            training=augment
        )

        self.class_counts = dict(
            Counter(self.labels.tolist())
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        image = self.transform(image)

        return image, label


# ==========================================
# DATALOADER FACTORY
# ==========================================

def create_dataloaders(
    model_name,
    batch_size=16,
    seed=42
):

    torch.manual_seed(seed)

    train_dataset = FaceOpsDataset(
        split="train",
        model_name=model_name,
        augment=True
    )

    val_dataset = FaceOpsDataset(
        split="val",
        model_name=model_name,
        augment=False
    )

    test_dataset = FaceOpsDataset(
        split="test",
        model_name=model_name,
        augment=False
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
