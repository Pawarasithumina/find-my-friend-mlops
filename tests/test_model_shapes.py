
import torch

from src.models.architectures import (
    create_model,
    CLASS_NAMES,
)

torch.set_num_threads(2)

MODELS = [
    "custom_cnn",
    "mobilenet_v3",
    "resnet18",
]


def main():
    print("\n===== FACEOPS MODEL VERIFICATION =====")

    # Simulated RGB input, 224x224
    sample = torch.randn(1, 3, 224, 224)

    for model_name in MODELS:
        print(f"\nTesting: {model_name}")

        # Avoid downloading pretrained weights
        # during architecture verification.
        model = create_model(
            model_name=model_name,
            num_classes=4,
            dropout=0.3,
            pretrained=False,
        )

        model.eval()

        with torch.no_grad():
            output = model(sample)

        assert output.shape == (1, 4), (
            f"Incorrect output: {output.shape}"
        )

        parameters = sum(
            p.numel() for p in model.parameters()
        )

        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print("Output:", output.shape)
        print(f"Parameters: {parameters:,}")
        print(f"Trainable: {trainable:,}")
        print("Status: PASSED")

    print("\nClasses:", CLASS_NAMES)
    print("\n===== ALL MODELS VERIFIED =====")


if __name__ == "__main__":
    main()
