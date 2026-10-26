from pathlib import Path
import torch
from torch import optim

from cnn_compression_toolkit.data import build_cifar10_bundle
from cnn_compression_toolkit.models.resnet import resnet56
from cnn_compression_toolkit.training.checkpoint import train_classifier
from cnn_compression_toolkit.evaluation.metrics import evaluate_classifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build data loaders (train / validation / calibration / test)
    bundle = build_cifar10_bundle(
        root="./data/cifar10",
        seed=20261024,
        calibration_size=512,
        validation_size=5000,
        batch_size=128,
        num_workers=2,
        download=True,  # set True to download CIFAR-10 if not present
    )

    # 2. Create the model
    model = resnet56(num_classes=10).to(device)

    # 3. Set up optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1.0e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 4. Train for 200 epochs (restores best validation-accuracy state)
    print("Starting training...")
    history = train_classifier(
        model=model,
        train_loader=bundle.train_loader,
        validation_loader=bundle.validation_loader,
        optimizer=optimizer,
        epochs=200,
        device=device,
        num_classes=10,
    )
    scheduler.step()  # optional: call per-epoch inside train_classifier if desired

    # 5. Evaluate on the held-out test set
    test_metrics = evaluate_classifier(
        model, bundle.test_loader, num_classes=10, device=device
    )
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1:  {test_metrics['macro_f1']:.4f}")

    # 6. Save the trained checkpoint
    ckpt_path = Path("./checkpoints/resnet56_cifar10_strong.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()