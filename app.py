import os

# 一定要放在 import torch / fastai 之前
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
import threading

# 修正 WindowsPath 問題
pathlib.WindowsPath = pathlib.PosixPath

# 限制 PyTorch CPU 使用量
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 關閉 mkldnn，避免某些 CPU 環境推論時 segfault
torch.backends.mkldnn.enabled = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

learn = None
model_error = None
predict_lock = threading.Lock()


def load_model():
    global learn, model_error

    if learn is not None:
        return learn

    try:
        print("========== Loading model ==========")
        print(f"MODEL_PATH: {MODEL_PATH}")
        print(f"MODEL_EXISTS: {MODEL_PATH.exists()}")
        print(f"Torch version: {torch.__version__}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")

        learn = load_learner(MODEL_PATH, cpu=True)
        learn.model.eval()

        print("Model loaded successfully.")
        model_error = None
        return learn

    except Exception as e:
        model_error = str(e)
        print("========== Failed to load model ==========")
        print(model_error)
        traceback.print_exc()
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            model = load_model()

            if model is None:
                error = "模型尚未成功載入。"
                if model_error:
                    error += f" 錯誤原因：{model_error}"

                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    error=error
                )

            file = request.files.get("image")

            if file is None or file.filename == "":
                error = "請先選擇一張圖片。"
                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    error=error
                )

            # 先用 PIL 開啟圖片，並轉成 RGB
            pil_img = Image.open(file.stream).convert("RGB")

            # 避免圖片過大造成 Render 記憶體壓力
            pil_img.thumbnail((1024, 1024))

            # 轉成 fastai 可讀格式
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            img = PILImage.create(buffer)

            # 避免多個請求同時推論
            with predict_lock:
                with torch.no_grad():
                    pred_class, pred_idx, probs = model.predict(img)

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
    model = load_model()

    if model is None:
        return {
            "status": "error",
            "message": "model not loaded",
            "model_exists": MODEL_PATH.exists(),
            "model_path": str(MODEL_PATH),
            "error": model_error
        }, 500

    return {
        "status": "ok",
        "message": "model loaded",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH)
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)