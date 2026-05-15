import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, request, render_template
from fastai.vision.all import *
from pathlib import Path
from PIL import Image
from io import BytesIO
import pathlib
import torch
import traceback
import numpy as np
import time

pathlib.WindowsPath = pathlib.PosixPath

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

print("========== Starting App ==========")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"MODEL_EXISTS: {MODEL_PATH.exists()}")
print(f"Torch version: {torch.__version__}")

learn = load_learner(MODEL_PATH, cpu=True)
learn.model.eval()

print("Model loaded successfully.")


def find_last_conv_layer(model):
    """
    自動尋找模型中最後一個 Conv2d layer。
    Grad-CAM 通常使用最後一層卷積層。
    """
    last_conv = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("找不到 Conv2d layer，無法產生 Grad-CAM。")

    return last_conv


def pil_to_tensor_for_model(pil_img):
    """
    將 PIL 圖片轉成模型可用 tensor。
    優先使用 fastai dataloader 的 after_item / after_batch 流程，
    讓 Grad-CAM 使用的前處理接近 learn.predict。
    """
    img = PILImage.create(pil_img)

    dl = learn.dls.test_dl([img], bs=1)
    batch = dl.one_batch()

    x = batch[0]

    return x


def make_gradcam(pil_img, class_idx, output_path):
    """
    產生 Grad-CAM 圖片並儲存。
    pil_img: 原始 PIL Image
    class_idx: 預測類別 index
    output_path: 輸出圖片路徑
    """
    model = learn.model
    model.eval()

    target_layer = find_last_conv_layer(model)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        x = pil_to_tensor_for_model(pil_img)

        # 確保在 CPU
        x = x.cpu()

        model.zero_grad()

        # Grad-CAM 需要 gradient，所以不能使用 torch.no_grad()
        output = model(x)

        if isinstance(output, tuple):
            output = output[0]

        score = output[0, int(class_idx)]
        score.backward()

        act = activations[0]      # shape: [1, C, H, W]
        grad = gradients[0]       # shape: [1, C, H, W]

        # Global Average Pooling gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)

        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize CAM to 0-1
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()

        if cam.max() != 0:
            cam = cam / cam.max()

        # Resize CAM to original image size
        cam_img = Image.fromarray(np.uint8(cam * 255)).resize(pil_img.size, Image.BILINEAR)
        cam_np = np.array(cam_img).astype(np.float32) / 255.0

        # 建立簡單 heatmap：紅色代表模型關注區域
        original = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0

        heatmap = np.zeros_like(original)
        heatmap[:, :, 0] = cam_np       # red channel
        heatmap[:, :, 1] = cam_np * 0.3 # slight yellow
        heatmap[:, :, 2] = 0

        overlay = original * 0.55 + heatmap * 0.45
        overlay = np.clip(overlay, 0, 1)

        result_img = Image.fromarray(np.uint8(overlay * 255))
        result_img.save(output_path)

    finally:
        forward_handle.remove()
        backward_handle.remove()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None
    gradcam_url = None

    if request.method == "POST":
        try:
            file = request.files.get("image")

            if file is None or file.filename == "":
                error = "請先選擇一張圖片。"
                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    error=error,
                    gradcam_url=gradcam_url
                )

            pil_img = Image.open(file.stream).convert("RGB")
            pil_img.thumbnail((224, 224))

            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)

            img = PILImage.create(buffer)

            with torch.no_grad():
                pred_class, pred_idx, probs = learn.predict(img)

            prediction = str(pred_class)
            confidence = f"{probs[pred_idx].item() * 100:.2f}%"

            # 產生 Grad-CAM
            timestamp = int(time.time())
            gradcam_filename = f"gradcam_{timestamp}.png"
            gradcam_path = STATIC_DIR / gradcam_filename

            make_gradcam(
                pil_img=pil_img,
                class_idx=int(pred_idx),
                output_path=gradcam_path
            )

            gradcam_url = f"/static/{gradcam_filename}"

        except Exception as e:
            error = f"圖片辨識或 Grad-CAM 產生時發生錯誤：{str(e)}"
            print("========== Prediction / Grad-CAM error ==========")
            print(e)
            traceback.print_exc()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        gradcam_url=gradcam_url
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "message": "app running",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH)
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
