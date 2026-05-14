from flask import Flask, request, render_template
from fastai.vision.all import *
from PIL import Image
import pathlib
import os

# 修正 Windows/Colab 匯出的 fastai model 在 Linux 讀取時的路徑問題
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)

learn = load_learner("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img = PILImage.create(file)
            pred_class, pred_idx, probs = learn.predict(img)

            prediction = str(pred_class)
            confidence = f"{probs[pred_idx].item() * 100:.2f}%"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)