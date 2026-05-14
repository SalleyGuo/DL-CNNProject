# =========================
# Render + FastAI Image Classifier
# app.py
# =========================

import os

# 限制 CPU threads，降低 Render 免費方案 worker 崩潰機率
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

# =========================
# 基本設定
# =========================

# 如果模型是在 Windows / Colab 環境匯出，Render Linux 讀取時可能需要這行
pathlib.WindowsPath = pathlib.PosixPath

# 強制使用 CPU
torch.set_num_threads(1)

app = Flask(__name__)

# 限制上傳檔案大小，例如 10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# 模型路徑：model.pkl 必須跟 app.py 放在同一層
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# =========================
# 載入模型
# =========================

learn = None

try:
    print(f"Loading model from: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    learn = load_learner(MODEL_PATH, cpu=True)

    print("Model loaded successfully.")

except Exception as e:
    print("Failed to load model.")
    print(e)
    traceback.print_exc()


# =========================
# 首頁與預測功能
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            if learn is None:
                error = "模型尚未成功載入，請檢查 Render logs。"
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

            # 讀取圖片
            img = PILImage.create(file)

            # 執行預測
            pred_class, pred_idx, probs = learn.predict(img)

            prediction = str(pred_class)
            confidence = f"{probs[pred_idx].item() * 100:.2f}%"

        except Exception as e:
            error = "圖片辨識時發生錯誤，請確認上傳的是圖片檔。"
            print("Prediction error:")
            print(e)
            traceback.print_exc()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


# =========================
# Render 健康檢查用
# =========================

@app.route("/health")
def health():
    if learn is None:
        return {"status": "error", "message": "model not loaded"}, 500

    return {"status": "ok", "message": "model loaded"}, 200


# =========================
# 本機測試用
# Render 正式部署會用 gunicorn，不會用這段啟動
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)