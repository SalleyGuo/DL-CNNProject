import os

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

print("========== Starting App ==========")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"MODEL_EXISTS: {MODEL_PATH.exists()}")
print(f"Torch version: {torch.__version__}")

learn = load_learner(MODEL_PATH, cpu=True)
learn.model.eval()

print("Model loaded successfully.")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            file = request.files.get("image")

            if file is None or file.filename == "":
                error = "請先選擇一張圖片。"
                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    error=error
                )

            pil_img = Image.open(file.stream).convert("RGB")
            pil_img.thumbnail((1024, 1024))

            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)

            img = PILImage.create(buffer)

            with torch.no_grad():
                pred_class, pred_idx, probs = learn.predict(img)

            prediction = str(pred_class)
            confidence = f"{probs[pred_idx].item() * 100:.2f}%"

        except Exception as e:
            error = f"圖片辨識時發生錯誤：{str(e)}"
            print("========== Prediction error ==========")
            print(e)
            traceback.print_exc()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
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