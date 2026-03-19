import numpy as np
import tensorflow as tf
from preprocess import preprocess_image

# ── Model registry ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "apple":   "model_weights/apple_model.keras",
    "potato":  "model_weights/potato_model.keras",
    "tomato":  "model_weights/tomato_model.keras",
}

CLASS_NAMES = {
    "apple": [
        "Apple_scab",
        "Apple_black_rot",
        "Apple_rust",
        "Apple_healthy"
    ],
 
    "potato": [
        "Potato_early_blight",
        "Potato_late_blight",
        "Potato_healthy"
    ],
    "tomato": [
        "Tomato_bacterial_spot",
        "Tomato_Septoria_leaf_spot",
        "Tomato_late_blight",
        "Tomato_healthy"
    ],
}

# Display names for UI — clean readable names without underscores
DISPLAY_NAMES = {
    "apple": [
        "Apple Scab",
        "Apple Black Rot",
        "Apple Cedar Rust",
        "Healthy"
    ],
    "potato": [
        "Early Blight",
        "Late Blight",
        "Healthy"
    ],
    "tomato": [
        "Bacterial Spot",
        "Tomato_Septoria_leaf_spot",
        "Late Blight",
        "Healthy"
    ],
}

# ── Global state ───────────────────────────────────────────────
current_model      = None
current_crop       = None
preprocessed_image = None


# ── Load model ─────────────────────────────────────────────────
def load_model(crop: str) -> dict:
    global current_model, current_crop

    if crop not in MODEL_REGISTRY:
        return {"error": f"No model found for crop: {crop}"}

    if current_crop == crop and current_model is not None:
        return {"message": f"{crop} model already loaded", "crop": crop}

    import os
    model_path = MODEL_REGISTRY[crop]
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}

    print(f"Loading {crop} model...")
    current_model = tf.keras.models.load_model(model_path)
    current_crop  = crop
    print(f"{crop} model loaded successfully")

    return {"message": f"{crop} model loaded successfully", "crop": crop}


# ── Store preprocessed image ───────────────────────────────────
def store_preprocessed(image_bytes: bytes) -> dict:
    global preprocessed_image
    preprocessed_image = preprocess_image(image_bytes)
    return {
        "message": "Image preprocessed successfully",
        "shape":   list(preprocessed_image.shape)
    }


# ── Run inference ──────────────────────────────────────────────
def run_inference() -> dict:
    global current_model, current_crop, preprocessed_image

    if current_model is None:
        return {"error": "No model loaded. Call /load-model first."}
    if preprocessed_image is None:
        return {"error": "No image found. Call /preprocess first."}

    preds   = current_model.predict(preprocessed_image, verbose=0)[0]
    top_idx = int(np.argmax(preds))

    classes      = CLASS_NAMES[current_crop]
    display      = DISPLAY_NAMES[current_crop]

    return {
        "crop":         current_crop,
        "disease":      classes[top_idx],
        "disease_label":display[top_idx],   # clean name for UI
        "confidence":   float(np.max(preds)),
        "all_scores": {
            display[i]: float(preds[i]) for i in range(len(classes))
        }
    }


# ── Get current state ──────────────────────────────────────────
def get_status() -> dict:
    return {
        "model_loaded": current_model is not None,
        "current_crop": current_crop,
        "image_ready":  preprocessed_image is not None
    }