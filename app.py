import os

# 限制 CPU threads，降低 Render worker 崩潰機率
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, request, render_template
from fastai.vision.all import *
from pathlib import Path
import pathlib
import torch
import traceback

# 修正 fastai 在 Linux 讀取 WindowsPath 的問題
pathlib.WindowsPath = pathlib.PosixPath

# 強制 CPU 推論
torch.set_num_threads(1)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

learn = None
model_error = None

print("========== App starting ==========")
print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"MODEL_EXISTS: {MODEL_PATH.exists()}")
print(f"Torch version: {torch.__version__}")

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")

    print("Start loading model...")
    learn = load_learner(MODEL_PATH, cpu=True)
    print("Model loaded successfully.")

except Exception as e:
    model_error = str(e)
    print("========== Failed to load model ==========")
    print(model_error)
    traceback.print_exc()


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            if learn is None:
                error = "模型尚未成功載入，請檢查 Render logs。"
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

            img = PILImage.create(file)

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
    if learn is None:
        return {
            "status": "error",
            "message": "model not loaded",
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "error": model_error
        }, 500

    return {
        "status": "ok",
        "message": "model loaded",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)