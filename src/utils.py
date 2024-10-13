import cv2
from skimage.feature import hog

def extract_features(image):
    img_resized = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features
