
import torch
import torch.nn as nn

from unet_model import UNet


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda:0")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Input size: 384 x 384")
    print("Batch size: 1")

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    images = torch.rand(1, 1, 384, 384, device=device)
    masks = (
        torch.rand(1, 1, 384, 384, device=device) > 0.95
    ).float()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(images)

    assert logits.shape == masks.shape, (
        f"Shape mismatch: logits={logits.shape}, masks={masks.shape}"
    )

    loss = criterion(logits, masks)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()

    print("Output shape:", tuple(logits.shape))
    print("Loss:", round(loss.item(), 4))

    try:
        peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        print("Peak allocated GPU memory:", round(peak_mb, 2), "MB")
    except RuntimeError as exc:
        print("GPU memory statistic unavailable:", exc)

    print("384x384 forward/backward/optimizer test PASSED.")


if __name__ == "__main__":
    main()
