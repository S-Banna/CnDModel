import argparse
import os

import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp


def predict(model, device, pre, post):
    stacked = np.concatenate([pre, post], axis=2)

    tensor = (
        torch.tensor(stacked)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)

    probs = (
        torch.sigmoid(logits)
        .cpu()
        .squeeze()
        .numpy()
    )

    return probs


def load_image(path):
    return np.array(
        Image.open(path).convert("RGB")
    ).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", help="e.g. mexico-earthquake_00000129")
    ap.add_argument("--samples", default="./samples")
    ap.add_argument("--model", default="../src/segmentation/model.pth")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=6,
        classes=1,
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.model,
            map_location=device
        )
    )

    model.eval()

    samples = args.samples
    pre_path = os.path.join(samples, f"{args.sample}_pre_disaster.png")
    post_path = os.path.join(samples, f"{args.sample}_post_disaster.png")
    for p in (pre_path, post_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    pre = load_image(pre_path)
    post = load_image(post_path)

    probs = predict(
        model,
        device,
        pre,
        post
    )

    binary = (
        probs > args.threshold
    ).astype(np.uint8)

    mask_visual = (binary * 255).astype(np.uint8)
    mask_path = os.path.join(samples, f"{args.sample}_building_mask.png")
    Image.fromarray(mask_visual).save(mask_path)
    print(f"Saved {mask_path}")


if __name__ == "__main__":
    main()