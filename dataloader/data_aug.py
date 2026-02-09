import numpy as np

def random_horizontal_flip(image, p=0.5):
    if np.random.rand() < p:
        image = np.fliplr(image)
    return image


def random_crop(image, crop_ratio=0.9):
    h, w, _ = image.shape
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)

    y = np.random.randint(0, h - ch + 1)
    x = np.random.randint(0, w - cw + 1)

    image = image[y:y+ch, x:x+cw]
    return image


def color_jitter(image, brightness=0.1, contrast=0.1):
    # brightness
    if brightness > 0:
        factor = 1.0 + np.random.uniform(-brightness, brightness)
        image = image * factor

    # contrast
    if contrast > 0:
        mean = image.mean(axis=(0, 1), keepdims=True)
        factor = 1.0 + np.random.uniform(-contrast, contrast)
        image = (image - mean) * factor + mean

    return np.clip(image, 0.0, 1.0)