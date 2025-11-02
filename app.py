from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms, models
from PIL import Image
import sqlite3

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----- 모델 로딩 -----
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# ----- 이미지 전처리 -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----- 결과 DB 저장 함수 -----
def save_prediction_to_db(filename, prediction):
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (filename, prediction) VALUES (?, ?)", (filename, prediction))
    conn.commit()
    conn.close()

# ----- 루트 페이지: 업로드 폼 -----
@app.route("/")
def index():
    return render_template("index.html")

# ----- 예측 처리 -----
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "파일이 포함되지 않았습니다.", 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return "선택된 파일이 없습니다.", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # 이미지 로드 및 전처리
    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # 예측
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = "실사 이미지" if predicted.item() == 0 else "애니메이션 이미지"

    # 결과 DB에 저장
    save_prediction_to_db(filename, label)

    return render_template("result.html", label=label, image_file=filename)

if __name__ == "__main__":
    app.run(debug=True)
