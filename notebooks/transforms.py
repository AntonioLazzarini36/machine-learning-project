import cv2
import numpy as np

class FlattenImageTransform:
    """
    Resizes a grayscale image and flattens it into a 1D feature vector.
    
    Args:
        target_size (tuple): The desired size of the image (height, width).
    """
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def __call__(self, sample):
        image = sample['image']  # Shape: (1, H, W)
        label = sample['label']

        # If image is a torch tensor, convert to numpy
        if hasattr(image, 'numpy'):
            image = image.squeeze(0).numpy()

        # Resize image
        image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

        # Flatten to 1D feature vector
        features = image_resized.flatten()

        return {'features': features, 'label': label}
    

# --- If you want to try HOG features instead of flattening ---
# You'll need scikit-image for HOG
# from skimage.feature import hog
# from skimage import exposure

# class HOGFeatureTransform:
#     def __init__(self, target_size=(64, 64), pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
#         self.target_size = target_size
#         self.pixels_per_cell = pixels_per_cell
#         self.cells_per_block = cells_per_block

#     def __call__(self, sample):
#         image = sample['image'] # Shape (1, H, W)
#         label = sample['label']

#         if isinstance(image, torch.Tensor):
#             image = image.squeeze(0).numpy()

#         image_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

#         # Calculate HOG features
#         # 'feature_vector=True' returns a 1D array of features
#         features = hog(image_resized, orientations=9, pixels_per_cell=self.pixels_per_cell,
#                        cells_per_block=self.cells_per_block, visualize=False, feature_vector=True,
#                        block_norm='L2-Hys') # L2-Hys is common for HOG

#         return {'features': features, 'label': label}

# --- Apply the transform to your dataset and create new DataLoaders ---
# Assuming 'dataset' is your original SignLangDataset instance
# You need to ensure your SignLangDataset's __getitem__ returns an appropriate format
# If your __getitem__ already returns a dict {'image': numpy_array, 'label': int},
# then this transform can directly be applied.