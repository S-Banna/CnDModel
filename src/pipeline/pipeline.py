import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
from config import RUN_LIST

# image loader helpers
def load_image(path):
    return np.array(
        Image.open(path).convert("RGB")
    ).astype(np.float32) / 255.0

def load_mask(path):

    if path is None:
        return None

    if not os.path.exists(path):
        return None

    return np.array(
        Image.open(path).convert("L")
    )

truth = None # if parsing fails, ground truth is optional

# model loader
def load_model():

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
            "../segmentation/model.pth",
            map_location=device
        )
    )

    model.eval()

    return model, device

# prediciton function
def predict(model, device, pre, post):

    stacked = np.concatenate(
        [pre, post],
        axis=2
    )

    tensor = (
        torch.tensor(stacked)
        .permute(2,0,1)
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

# damage overlay (mask overlayed to post)
def damage_overlay(post, binary):

    out = post.copy()

    out[binary == 1] = [1.0, 0.2, 0.2]

    return out

# border overlay (mask borders overlayed to post)
def border_overlay(post, binary):

    mask = (binary * 255).astype(np.uint8)

    border = mask - cv2.erode(
        mask,
        np.ones((3,3), np.uint8),
        iterations=3
    )

    out = post.copy()

    out[border > 0] = [1.0, 0.2, 0.2]

    return out

# main run function
def run_pipeline(
    pre_path,
    post_path,
    mask_path=None,
    threshold=0.5
):

    print(f"\nProcessing: {post_path}")

    pre = load_image(pre_path)
    post = load_image(post_path)
    truth = load_mask(mask_path)

    model, device = load_model()

    probs = predict(
        model,
        device,
        pre,
        post
    )

    binary = (
        probs > threshold
    ).astype(np.uint8)

    overlay = damage_overlay(
        post,
        binary
    )

    border = border_overlay(
        post,
        binary
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    axes[0,0].imshow(pre)
    axes[0,0].set_title("Pre")

    axes[0,1].imshow(post)
    axes[0,1].set_title("Post")

    axes[0,2].imshow(overlay)
    axes[0,2].set_title("Damage Overlay")

    if truth is not None:
        axes[1,0].imshow(
            truth,
            cmap="gray"
        )
    else:
        axes[1,0].imshow(
            np.zeros(
                binary.shape,
                dtype=np.uint8
            ),
            cmap="gray"
        )
        h, w = binary.shape

        axes[1,0].plot(
            [0, w],
            [0, h],
            "r",
            linewidth=4
        )
        axes[1,0].plot(
            [0, w],
            [h, 0],
            "r",
            linewidth=4
        )

    axes[1,0].set_title(
        "Ground Truth"
    )

    axes[1,1].imshow(binary, cmap="gray")
    axes[1,1].set_title("Prediction")

    axes[1,2].imshow(border)
    axes[1,2].set_title("Border Overlay")

    for ax in axes.flatten():
        ax.axis("off")

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    save_name = os.path.splitext(
        os.path.basename(post_path)
    )[0]

    fig.savefig(
        f"outputs/{save_name}_pipeline.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved outputs/{save_name}_pipeline.png"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pre")
    parser.add_argument("--post")
    parser.add_argument(
        "--mask"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None
    )

    parser.add_argument(
        "--batch",
        action="store_true"
    )

    args = parser.parse_args()

    if args.threshold is None:

        threshold = 0.5

        print(
            "Threshold not selected. "
            "Using default threshold = 0.5"
        )

    else:
        threshold = args.threshold

    if args.batch:

        for sample in RUN_LIST:

            run_pipeline(
                sample["pre"],
                sample["post"],
                sample["mask"],
                threshold
            )

    else:

        run_pipeline(
            args.pre,
            args.post,
            args.mask,
            threshold
        )


if __name__ == "__main__":
    main()