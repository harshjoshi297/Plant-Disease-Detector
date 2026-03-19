# 🌿 Plant Disease Detection & Advisory System

An end-to-end plant disease detection system that classifies diseases from leaf images using ResNet50 transfer learning and provides crop-specific treatment advice via an LLM-powered chat assistant.

## 🔗 Links
- **GitHub**: [Plant Disease Detector](https://github.com/harshjoshi297/Plant-Disease-Detector)

---

## 🧠 How It Works

1. User selects a crop (Apple, Potato, Tomato)
2. Uploads or captures a leaf photo
3. ResNet50 model classifies the disease
4. Confidence scores shown for all classes
5. LLM chat assistant activated for treatment advice

---

## 🌱 Supported Crops & Diseases

| Crop | Diseases |
|------|----------|
| Apple | Scab, Black Rot, Cedar Rust, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Septoria Leaf Spot, Late Blight, Healthy |

---

## 📊 Model Performance

| Crop | Test Accuracy | Dataset Size |
|------|--------------|--------------|
| Apple | 77% | 18,613 images |
| Potato | 73.9% | 4,595 images |
| Tomato | 75% | 21,288 images |

### Training Strategy
- **Phase 1**: Frozen ResNet50 base, only custom classification head trained
- **Phase 2**: Top 15 ResNet50 layers unfrozen with BatchNormalization layers kept frozen to prevent model collapse
- **Class imbalance** handled using sklearn computed class weights

---

## 🛠️ Tech Stack

**Backend**
- FastAPI — REST API framework
- TensorFlow / Keras — ResNet50 transfer learning
- LangChain — ChatPromptTemplate for structured prompts
- Groq — LLaMA 3.3 70B inference
- Pydantic — request/response validation
- Docker — containerization

**Frontend**
- Gradio — interactive web UI
- Webcam and image upload support

---

## 🏗️ Architecture
```
User (Gradio UI)
      ↓
FastAPI Backend
  ├── /load-model   → loads crop-specific ResNet50 model
  ├── /preprocess   → resize + normalize image
  ├── /predict      → run inference, return disease + confidence
  └── /chat         → LangChain + Groq LLM response
```

---

## 🚀 Run with Docker (Recommended)

### Prerequisites
- Docker
- Docker Compose

### Steps
```bash
# Clone the repo
git clone https://github.com/harshjoshi297/Plant-Disease-Detector.git
cd Plant-Disease-Detector

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Add model weights to backend/model_weights/
# apple_model.keras, potato_model.keras, tomato_model.keras

# Build and run
docker-compose up --build
```

Open `http://localhost:7860` for the UI and `http://localhost:8000/docs` for the API.

---

## 💻 Run Locally

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
source ../venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

---

## 📁 Project Structure
```
Plant-Disease-Detector/
├── backend/
│   ├── main.py          # FastAPI endpoints
│   ├── model.py         # model loading and inference
│   ├── preprocess.py    # image preprocessing
│   ├── llm.py           # LangChain + Groq chat
│   ├── schemas.py       # Pydantic schemas
│   ├── requirements.txt
│   ├── Dockerfile
│   └── model_weights/   # .keras model files (not in repo)
├── frontend/
│   ├── app.py           # Gradio UI
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── .env                 # GROQ_API_KEY (not in repo)
```

---

## ⚠️ Note on Model Weights

Model weights (~100MB each) are not included in this repository. To use the app:
1. Train your own models using the training notebooks
2. Or contact me for access to pre-trained weights

---

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/load-model` | Load crop-specific model |
| POST | `/preprocess` | Preprocess uploaded image |
| POST | `/predict` | Run disease classification |
| POST | `/chat` | Get LLM treatment advice |
| GET | `/status` | Check model and image state |
| GET | `/docs` | Swagger UI |
