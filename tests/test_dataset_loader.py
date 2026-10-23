
import torch

from src.data.dataset_loader import (
    create_dataloaders,
    FaceOpsDataset,
    CLASS_NAMES
)


MODELS = [
    "custom_cnn",
    "mobilenet_v3",
    "resnet18"
]

EXPECTED_SPLITS = {
    "train": 144,
    "val": 32,
    "test": 33
}


def main():

    print("\n===== FACEOPS DATALOADER TEST =====")

    for model_name in MODELS:

        print(f"\nMODEL: {model_name}")

        train_loader, val_loader, test_loader = (
            create_dataloaders(
                model_name=model_name,
                batch_size=8
            )
        )

        loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

        for split, loader in loaders.items():

            assert len(loader.dataset) == (
                EXPECTED_SPLITS[split]
            )

            images, labels = next(iter(loader))

            assert images.ndim == 4
            assert images.shape[1:] == (3, 224, 224)
            assert images.dtype == torch.float32
            assert labels.dtype == torch.int64

            assert torch.isfinite(images).all()

            assert labels.min() >= 0
            assert labels.max() < 4

            print(
                f"{split.upper():5s} | "
                f"Samples: {len(loader.dataset):3d} | "
                f"Batch: {list(images.shape)} | "
                f"Status: PASSED"
            )

        # Check that validation is deterministic.
        val_dataset = FaceOpsDataset(
            split="val",
            model_name=model_name,
            augment=False
        )

        image_1, _ = val_dataset[0]
        image_2, _ = val_dataset[0]

        assert torch.equal(image_1, image_2)

        print("Validation determinism: PASSED")

    print("\nClass names:", CLASS_NAMES)

    print("\n===== ALL DATALOADER TESTS PASSED =====")


if __name__ == "__main__":
    main()
