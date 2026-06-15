import os
import sys
import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.transformer import TransformerModel
from models.crnn import CRNNModel

CLASSES = ["normal", "swerving", "tailgating"]
NUM_FRAMES = 32
IMG_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def load_model(model_choice: str):
    if model_choice == "Transformer":
        model = TransformerModel()
        ckpt = os.path.join(BASE_DIR, "models", "transformer.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(BASE_DIR, "models", "transformer_best.pt")
    else:
        model = CRNNModel()
        ckpt = os.path.join(BASE_DIR, "models", "crnn.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(BASE_DIR, "models", "crnn_best.pt")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model file not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def preprocess_frames(images: list) -> torch.Tensor:
    processed = []
    if images is None:
        return torch.zeros(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)

    for item in images:
        if isinstance(item, (tuple, list)):
            img = item[0]
        elif isinstance(item, dict):
            img = item.get("image") or item.get("name")
        elif hasattr(item, "image"):
            img = item.image
        else:
            img = item

        if isinstance(img, str):
            try:
                img = Image.open(img)
            except Exception:
                continue

        if img is not None:
            img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            processed.append(norm(img))

    total = len(processed)
    if total == 0:
        return torch.zeros(1, NUM_FRAMES, 3, IMG_SIZE, IMG_SIZE)

    if total < NUM_FRAMES:
        processed += [processed[-1]] * (NUM_FRAMES - total)
    elif total > NUM_FRAMES:
        step = total / NUM_FRAMES
        indices = [int(i * step) for i in range(NUM_FRAMES)]
        processed = [processed[i] for i in indices]

    return torch.stack(processed).unsqueeze(0)


def predict(images, model_choice):
    if not images:
        return "Upload frames to begin.", {}

    try:
        model = load_model(model_choice)
    except FileNotFoundError as e:
        return str(e), {}

    tensor = preprocess_frames(images).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    results = {cls: round(p, 4) for cls, p in zip(CLASSES, probs)}
    predicted = CLASSES[probs.index(max(probs))]

    label_map = {
        "normal": "Normal driving",
        "swerving": "Swerving detected",
        "tailgating": "Tailgating detected",
    }

    return label_map[predicted], results


CSS = """
:root {
    --c-blue: #2563eb;
    --c-light-blue: #eff6ff;
    --c-white: #ffffff;
    --c-black: #0f172a;
}

body, .gradio-container {
    background-color: var(--c-light-blue) !important;
    font-family: system-ui, -apple-system, sans-serif !important;
    color: var(--c-black) !important;
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

.app-header {
    background-color: var(--c-blue) !important;
    color: var(--c-white) !important;
    border-radius: 20px !important;
    padding: 30px !important;
    text-align: center !important;
    margin-bottom: 24px !important;
}

.custom-card {
    background-color: var(--c-white) !important;
    border: 2px solid var(--c-blue) !important;
    border-radius: 20px !important;
    padding: 24px !important;
}

.gr-input-label, label, .label-wrap span {
    color: var(--c-black) !important;
    font-weight: 600 !important;
}

fieldset legend, fieldset span {
    color: var(--c-white) !important;
}

.panel-title {
    font-size: 1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    color: var(--c-blue) !important;
    display: block !important;
    margin-bottom: 12px !important;
    border-bottom: 2px solid var(--c-light-blue) !important;
    padding-bottom: 12px !important;
}

[data-testid="file-upload"] {
    border: 2px dashed var(--c-blue) !important;
    background-color: var(--c-light-blue) !important;
    border-radius: 16px !important;
}

#run-btn {
    background-color: var(--c-blue) !important;
    color: var(--c-white) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-weight: 700 !important;
}

.result-box textarea {
    background-color: var(--c-light-blue) !important;
    border: 1px solid var(--c-blue) !important;
    border-radius: 12px !important;
    color: var(--c-black) !important;
}

.progress-bar > div, .bar, .fill {
    background-color: var(--c-blue) !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(title="Driving Behavior Analysis") as demo:

    gr.HTML("""
        <div class="app-header">
            <h1>Driving Behavior Analysis</h1>
            <p>Upload dashcam frames, pick a model, and run the analysis.</p>
        </div>
    """)

    with gr.Row(equal_height=True):

        with gr.Column(elem_classes=["custom-card"]):
            gr.HTML('<span class="panel-title">Input Configuration</span>')

            images_input = gr.File(
                label="Upload Frames",
                type="filepath",
                file_count="multiple"
            )

            model_choice = gr.Radio(
                choices=["Transformer", "CRNN"],
                value="Transformer",
                label="Select Model"
            )

            submit_btn = gr.Button("Run analysis", elem_id="run-btn")

        with gr.Column(elem_classes=["custom-card"]):
            gr.HTML('<span class="panel-title">Analysis Output</span>')

            prediction_out = gr.Textbox(
                label="Result",
                interactive=False,
                placeholder="Waiting...",
                elem_classes=["result-box"]
            )

            confidence_out = gr.Label(
                label="Confidence",
                num_top_classes=3
            )

    submit_btn.click(
        fn=predict,
        inputs=[images_input, model_choice],
        outputs=[prediction_out, confidence_out],
    )

if __name__ == "__main__":
    demo.launch(css=CSS)