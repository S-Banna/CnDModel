//Patch Creation if the images are larger than 512×512

def create_patches(image_path, patch_size=512, stride=256):
    image = Image.open(image_path)
    width, height = image.size
    patches = []
    for x in range(0, width - patch_size + 1, stride):
        for y in range(0, height - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    return patches
