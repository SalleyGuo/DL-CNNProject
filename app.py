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

CODE_VERSION = "stable-activation-heatmap-v1"

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

print("========== Starting App ==========", flush=True)
print(f"CODE_VERSION: {CODE_VERSION}", flush=True)
print(f"MODEL_PATH: {MODEL_PATH}", flush=True)
print(f"MODEL_EXISTS: {MODEL_PATH.exists()}", flush=True)
print(f"Torch version: {torch.__version__}", flush=True)

learn = load_learner(MODEL_PATH, cpu=True)
learn.model.eval()

print("Model loaded successfully.", flush=True)


def log_step(message):
    print(f"[DEBUG] {message}", flush=True)
    sys.stdout.flush()


def find_last_conv_layer(model):
    last_conv = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("找不到 Conv2d layer，無法產生熱區圖。")

    return last_conv


def pil_to_fastai_img(pil_img):
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return PILImage.create(buffer)


def make_activation_heatmap(pil_img):
    """
    穩定版熱區圖：
    不做 backward，不做真正 Grad-CAM。
    只抓最後一層卷積層 activation，產生模型關注區域的近似熱區圖。
    這版適合 Render 免費方案，較不容易 500 或 SIGSEGV。
    """

    log_step("Activation heatmap started")

    model = learn.model
    model.eval()

    target_layer = find_last_conv_layer(model)
    log_step(f"Target layer found: {target_layer}")

    activation_holder = {}

    def forward_hook(module, input, output):
        activation_holder["activation"] = output.detach()

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        img = pil_to_fastai_img(pil_img)

        # 先正常預測，這部分你原本已經成功
        with torch.no_grad():
            pred_class, pred_idx, probs = learn.predict(img)

        prediction = str(pred_class)
        confidence_value = float(probs[pred_idx].item())

        log_step(f"Prediction done: {prediction}, confidence={confidence_value}")

        # 為了取得 activation，再做一次 forward
        dl = learn.dls.test_dl([img], bs=1)
        xb = dl.one_batch()[0].cpu()

        with torch.no_grad():
            output = model(xb)

        if "activation" not in activation_holder:
            raise ValueError("沒有取得 activation，無法產生熱區圖。")

        activation = activation_holder["activation"]

        # activation shape: [1, C, H, W]
        # 簡化方式：對 channel 取平均，形成 heatmap
        cam = activation.mean(dim=1).squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

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

        log_step("Activation heatmap created successfully")

        return prediction, confidence_value, result_img

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
            log_step("POST request received")

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

            prediction, conf_value, heatmap_img = make_activation_heatmap(pil_img)

            confidence = f"{conf_value * 100:.2f}%"

            gradcam_filename = f"gradcam_{int(time.time())}.png"
            gradcam_path = STATIC_DIR / gradcam_filename
            heatmap_img.save(gradcam_path)

            gradcam_url = f"/static/{gradcam_filename}"

            log_step("POST request finished successfully")

        except Exception as e:
            error = f"圖片辨識或熱區圖產生時發生錯誤：{str(e)}"
            print("========== Prediction / Heatmap error ==========", flush=True)
            print(e, flush=True)
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
        "code_version": CODE_VERSION,
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH)
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
