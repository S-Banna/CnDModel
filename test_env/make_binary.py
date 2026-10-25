import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

def predict(model, device, pre, post):
    stacked = np.concatenate(
        [pre, post],
        axis=2
    )

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
            "../src/segmentation/model.pth",
            map_location=device
        )
    )

    model.eval()

    threshold = 0.5
    pre = load_image("mexico-earthquake_00000129_pre_disaster.png")
    post = load_image("mexico-earthquake_00000129_post_disaster.png")

    probs = predict(
        model,
        device,
        pre,
        post
    )

    binary = (
        probs > threshold
    ).astype(np.uint8)

    mask_visual = (binary * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_visual)
    mask_image.save("mexico-earthquake_00000129_building_mask.png")

    mask        = np.array(Image.open("mexico-earthquake_00000129_post_disaster_rgb.png"))
    binary_mask = np.isin(mask, [3, 4]).astype(np.float32)
    binary_mask_visual = (binary_mask * 255).astype(np.uint8)
    binary_image = Image.fromarray(binary_mask_visual)
    binary_image.save("truth.png")

if __name__ == "__main__":
    main()