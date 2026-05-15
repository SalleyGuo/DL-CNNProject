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


def log_step(message):
    print(f"[DEBUG] {message}", flush=True)
    sys.stdout.flush()


def make_gradcam_and_predict(pil_img):
    """
    同時完成：
    1. 模型預測
    2. Grad-CAM 產生

    加入 debug log，方便從 Render logs 判斷卡在哪一步。
    """

    log_step("Grad-CAM started")

    model = learn.model
    model.eval()

    log_step("Finding last conv layer")
    target_layer = find_last_conv_layer(model)
    log_step(f"Target layer found: {target_layer}")

    activation_holder = {}

    def forward_hook(module, input, output):
        log_step("Forward hook triggered")
        activation_holder["activation"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        log_step("Creating fastai batch")
        xb = pil_to_batch(pil_img)
        xb = xb.cpu()
        log_step(f"Batch created: {xb.shape}")

        model.zero_grad(set_to_none=True)

        log_step("Forward pass started")
        output = model(xb)
        log_step("Forward pass finished")

        if isinstance(output, tuple):
            output = output[0]

        log_step(f"Output shape: {output.shape}")

        probs = torch.softmax(output, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

        log_step(f"Predicted index: {pred_idx}")
        log_step(f"Confidence: {confidence}")

        score = output[0, pred_idx]

        log_step("Backward pass started")
        score.backward()
        log_step("Backward pass finished")

        if "activation" not in activation_holder:
            raise ValueError("沒有取得 activation，Grad-CAM 無法產生。")

        activation = activation_holder["activation"]
        gradients = activation.grad

        if gradients is None:
            raise ValueError("沒有取得 gradients，Grad-CAM 無法產生。")

        log_step(f"Activation shape: {activation.shape}")
        log_step(f"Gradient shape: {gradients.shape}")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        log_step("CAM calculated")

        cam_img = Image.fromarray(np.uint8(cam * 255)).resize(
            pil_img.size,
            Image.BILINEAR
        )

        cam_np = np.array(cam_img).astype(np.float32) / 255.0
        original = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0

        heatmap = np.zeros_like(original)
        heatmap[:, :, 0] = cam_np
        heatmap[:, :, 1] = cam_np * 0.25
        heatmap[:, :, 2] = 0

        overlay = original * 0.60 + heatmap * 0.40
        overlay = np.clip(overlay, 0, 1)

        result_img = Image.fromarray(np.uint8(overlay * 255))

        try:
            pred_class = str(learn.dls.vocab[pred_idx])
        except Exception:
            pred_class = str(pred_idx)

        log_step("Grad-CAM image created successfully")

        return pred_class, pred_idx, confidence, result_img

    finally:
        handle.remove()
        log_step("Hook removed")


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

            pred_class, pred_idx, conf_value, gradcam_img = make_gradcam_and_predict(pil_img)

            prediction = str(pred_class)
            confidence = f"{conf_value * 100:.2f}%"
            
            gradcam_filename = f"gradcam_{int(time.time())}.png"
            gradcam_path = STATIC_DIR / gradcam_filename
            gradcam_img.save(gradcam_path)
            
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
