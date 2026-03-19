import numpy as np
from PIL import Image
import io

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Open image from raw bytes
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB — handles PNG with alpha channel (RGBA), 
    # grayscale, or any other format
    img = img.convert("RGB")
    
    # Resize to 224x224 — what ResNet50 expects
    img = img.resize(IMG_SIZE)
    
    # Convert to numpy array and normalize to [0, 1]
    arr = np.array(img) / 255.0
    
    # Add batch dimension — model expects (1, 224, 224, 3) not (224, 224, 3)
    arr = np.expand_dims(arr, axis=0)
    
    return arr  # shape: (1, 224, 224, 3)


def validate_image(image_bytes: bytes) -> bool:
    # Check the file is actually a valid image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # checks file integrity without fully loading it
        return True
    except Exception:
        return False
