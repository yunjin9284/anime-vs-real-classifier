# anime vs Real 이미지 분류 웹 서비스

## 프로젝트 개요

웹에서 업로드된 이미지를 딥러닝 모델 (MobileNetV2)을 통해  
**"만화(anime)"**인지 **"실사(real)"**인지 분류하는 Flask 기반 웹 애플리케이션입니다.

---

## 폴더 구조

```
Project_2570332/
├── app.py               # 웹 애플리케이션 실행 파일
├── train_model.py       # 모델 학습 스크립트
├── model.pth            # 학습된 모델 저장 파일
├── requirements.txt     # 필요한 라이브러리 목록
├── results.db           # SQLite 결과 저장 DB
├── templates/           # HTML 템플릿 폴더
│   ├── index.html
│   └── result.html
├── static/uploads/      # 업로드된 이미지 저장 폴더
├── dataset/             # 학습/검증용 이미지 데이터셋
│   ├── train/anime/
│   ├── train/real/
│   ├── val/anime/
│   └── val/real/
```

---

## 실행 방법

```bash
# 1. 가상환경 실행 (Windows 기준)
.venv\Scripts\activate

# 2. 라이브러리 설치
pip install -r requirements.txt

# 3. 모델 학습 (선택)
python train_model.py

# 4. 웹 서버 실행
python app.py
```

웹 브라우저에서 아래 주소 접속:  
 http://127.0.0.1:5000

---

## 사용 기술

- Python 3
- Flask (웹 프레임워크)
- PyTorch (딥러닝, MobileNetV2)
- SQLite (예측 결과 저장)
- HTML / CSS (템플릿)

---

## 시연 영상

YouTube 링크: _(https://youtu.be/DI419gLgVTQ)_

---

## 링크

https://github.com/yunjin9284/anime-vs-real-classifier
