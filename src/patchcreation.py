# Patch Creation if the images are larger than 512×512

from PIL import Image
import os

def create_patches(image_path, patch_size=512, stride=256, save_dir=None):
    """
    Splits an image into overlapping patches of a given size and stride.
    
    Args:
        image_path (str): Path to the input image file.
        patch_size (int): Size (in pixels) of each square patch.
        stride (int): Step (in pixels) to move the window for the next patch.
        save_dir (str): Optional directory to save patches. If None, only returns them.
    
    Returns:
        list: List of PIL Image objects representing the patches.
    """
    image = Image.open(image_path)
    width, height = image.size
    patches = []

    # Create output folder if saving
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    patch_id = 0
    for x in range(0, width - patch_size + 1, stride):
        for y in range(0, height - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

            if save_dir:
                patch_filename = f"patch_{patch_id:04d}.png"
                patch.save(os.path.join(save_dir, patch_filename))
            patch_id += 1

    return patches