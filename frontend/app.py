import gradio as gr
import requests
from PIL import Image
import io

import os
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

current_prediction = {}

def load_crop_model(crop):
    if not crop:
        return "Please select a crop first."
    try:
        res = requests.post(
            f"{BACKEND}/load-model",
            data={"crop": crop.lower()}
        )
        if res.status_code == 200:
            return f"{crop} model loaded successfully"
        else:
            return f"Error: {res.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"Could not connect to backend: {str(e)}"


def analyse_image(crop, image):
    global current_prediction

    if not crop:
        return "Please select a crop first.", []
    if image is None:
        return "Please upload an image first.", []

    try:
        pil_image = Image.fromarray(image)
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)

        res = requests.post(
            f"{BACKEND}/preprocess",
            files={"image": ("leaf.jpg", buf, "image/jpeg")}
        )
        if res.status_code != 200:
            return f"Preprocessing failed: {res.json().get('detail')}", []

        res = requests.post(f"{BACKEND}/predict")
        if res.status_code != 200:
            return f"Prediction failed: {res.json().get('detail')}", []

        pred = res.json()
        current_prediction = pred

        disease    = pred["disease_label"]
        confidence = pred["confidence"]
        all_scores = pred["all_scores"]
        is_healthy = "healthy" in pred["disease"].lower()

        if is_healthy:
            status = f"Plant looks healthy! ({confidence:.1%} confidence)"
        else:
            status = f"Detected: {disease} ({confidence:.1%} confidence)"

        scores_text = "\n".join([
            f"{cls}: {score:.1%}"
            for cls, score in sorted(
                all_scores.items(), key=lambda x: x[1], reverse=True
            )
        ])

        result = f"## {status}\n\n**All scores:**\n{scores_text}"

        initial_msg = get_initial_message(pred)
        initial_history = [
            {"role": "assistant", "content": initial_msg}
        ] if initial_msg else []

        return result, initial_history

    except Exception as e:
        return f"Error: {str(e)}", []


def get_initial_message(pred):
    if not pred:
        return None
    disease    = pred["disease_label"]
    confidence = pred["confidence"]
    crop       = pred["crop"]
    is_healthy = "healthy" in pred["disease"].lower()

    if is_healthy:
        return (
            f"Good news! Your {crop} plant appears healthy "
            f"({confidence:.1%} confidence). Ask me anything about "
            f"keeping it healthy or preventing diseases."
        )
    return (
        f"I've detected {disease} in your {crop} plant "
        f"({confidence:.1%} confidence). Ask me anything about "
        f"treatment, causes, or prevention."
    )


def extract_content(content):
    # Handle Gradio 6.0 content format
    # content can be a plain string or a list of dicts
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )
    return str(content)


def chat_response(message, history):
    global current_prediction

    if not current_prediction:
        return "Please analyse an image first to get a diagnosis."

    try:
        chat_history = []
        for msg in history:
            if isinstance(msg, dict):
                chat_history.append({
                    "role":    msg["role"],
                    "content": extract_content(msg.get("content", ""))
                })

        res = requests.post(f"{BACKEND}/chat", json={
            "crop":          current_prediction["crop"],
            "disease":       current_prediction["disease"],
            "disease_label": current_prediction["disease_label"],
            "confidence":    current_prediction["confidence"],
            "message":       message,
            "history":       chat_history
        })

        if res.status_code == 200:
            return res.json()["response"]
        return f"Chat error: {res.json().get('detail', 'Unknown error')}"

    except Exception as e:
        return f"Error: {str(e)}"


def handle_chat(message, history):
    if not message.strip():
        return history, ""
    response = chat_response(message, history)
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": response}
    ]
    return history, ""


# ── UI ─────────────────────────────────────────────────────────
with gr.Blocks() as app:

    gr.HTML("""
        <div style="text-align:center; padding:20px 0 10px 0">
            <h1 style="font-size:2.2em; color:#166534">🌿 Plant Disease Detector</h1>
            <p style="color:#64748b; font-size:1.1em">
                Select your crop, upload a leaf photo, and get an instant AI diagnosis
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1 — Select your crop")
            crop_dropdown = gr.Dropdown(
                choices     = ["Apple", "Potato", "Tomato"],
                label       = "Crop",
                value       = None,
                interactive = True
            )
            model_status = gr.Textbox(
                label       = "Model status",
                interactive = False,
                value       = "No model loaded"
            )

            gr.Markdown("### Step 2 — Upload leaf photo")
            image_input = gr.Image(
                label   = "Leaf photo",
                sources = ["upload", "webcam"],
                type    = "numpy"
            )
            analyse_btn = gr.Button(
                "Analyse",
                variant = "primary",
                size    = "lg"
            )

        with gr.Column(scale=1):
            gr.Markdown("### Diagnosis result")
            result_output = gr.Markdown(
                value = "Upload an image and click Analyse to see results."
            )

            gr.Markdown("### Step 3 — Chat with AI assistant")
            gr.Markdown(
                "Analyse an image first, then ask about treatment, "
                "causes, and prevention."
            )
            chatbot = gr.Chatbot(
                height = 350,
                label  = "AI Assistant"
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder = "Ask about treatment, prevention, causes...",
                    label       = "",
                    scale       = 4
                )
                chat_btn = gr.Button("Send", scale=1)

            gr.Examples(
                examples = [
                    "What causes this disease?",
                    "How do I treat it organically?",
                    "Will it spread to other plants?",
                    "What preventive steps should I take?",
                    "When should I consult an agronomist?"
                ],
                inputs = chat_input,
                label  = "Example questions"
            )

    # ── Event handlers ─────────────────────────────────────────
    crop_dropdown.change(
        fn      = load_crop_model,
        inputs  = crop_dropdown,
        outputs = model_status
    )

    analyse_btn.click(
        fn      = analyse_image,
        inputs  = [crop_dropdown, image_input],
        outputs = [result_output, chatbot]
    )

    chat_btn.click(
        fn      = handle_chat,
        inputs  = [chat_input, chatbot],
        outputs = [chatbot, chat_input]
    )

    chat_input.submit(
        fn      = handle_chat,
        inputs  = [chat_input, chatbot],
        outputs = [chatbot, chat_input]
    )

if __name__ == "__main__":
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        theme       = gr.themes.Soft(
            primary_hue   = "green",
            secondary_hue = "emerald",
            neutral_hue   = "slate"
        )
    )