# 🌊 Flood Risk Predictor — ML Opsidian: Genesis (Final Round)

End-to-end MLOps solution for flood risk scoring, built on the Initial Round ensemble model.

---

## 🗂️ Project Structure

```
flood-risk-mlops/
├── train.py                  # Training script (saves model + MLflow tracking)
├── app/
│   ├── predict.py            # Inference pipeline (feature engineering → ensemble → calibration)
│   ├── main.py               # FastAPI REST API
│   └── dashboard.py          # Streamlit UI (single predict + batch + monitoring)
├── model/                    # Saved model artifacts (created by train.py)
├── mlruns/                   # MLflow experiment tracking (auto-created)
├── .github/workflows/ci.yml  # GitHub Actions CI/CD
├── Dockerfile
├── render.yaml               # Render.com deployment config
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/<your-team>/<repo-name>.git
cd flood-risk-mlops
pip install -r requirements.txt
```

### 2. Train the model

Place `train.csv` and `test.csv` in the project root, then:

```bash
python train.py
```

This will:
- Run 10-fold CV training (LGB + CatBoost + XGBoost + ExtraTrees + MLP)
- Log all metrics and hyperparameters to **MLflow**
- Save `model/flood_model.pkl` (the full artifact bundle)
- Save `submission_v6.csv`

### 3. View MLflow experiment tracking

```bash
mlflow ui
# Open http://localhost:5000
```

You'll see every run with OOF RMSE per model, blend weights, and artifacts.

### 4. Run the Streamlit app locally

```bash
streamlit run app/dashboard.py
# Open http://localhost:8501
```

Three pages:
- **Single Prediction** — fill a form, get a risk score + gauge
- **Batch Prediction** — upload a CSV, download scored results
- **Monitoring** — charts of all prediction history

### 5. Run the FastAPI backend locally

```bash
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single-row JSON prediction |
| POST | `/predict/batch` | CSV upload batch prediction |
| GET | `/metrics` | Prediction log summary |
| GET | `/logs` | Recent prediction history |

---

## 🚀 Deployment (Render.com — Free)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-team>/<repo>.git
git push -u origin main
```

### Step 2 — Deploy on Render

1. Go to [render.com](https://render.com) → New → **Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` and auto-deploys both services

> **Important:** After deploying, copy your trained `model/flood_model.pkl` to the Render disk, or add model training as a build step.

### Step 3 — Set up CI/CD auto-deploy

1. In Render dashboard → your service → **Deploy Hook** → copy the URL
2. In GitHub → Settings → Secrets → add `RENDER_DEPLOY_HOOK` = that URL
3. Now every push to `main` runs tests → auto-deploys ✅

---

## 🧪 CI/CD Pipeline (GitHub Actions)

Every push to `main`:
1. **Lint** — flake8 checks for syntax errors
2. **Tests** — runs `tests/` (optional, won't fail CI if absent)
3. **Deploy** — triggers Render webhook to redeploy

---

## 🤖 MLOps Components Implemented

| Component | Implementation |
|-----------|---------------|
| Experiment tracking | MLflow — logs every run, params, metrics, artifacts |
| Model versioning | `model/flood_model.pkl` versioned via Git + MLflow run ID |
| Feature engineering | Encapsulated in `app/predict.py` (reusable pipeline) |
| Data validation | Graceful handling of missing columns & unknown categories |
| REST API | FastAPI with auto-generated OpenAPI docs |
| Web UI | Streamlit — single + batch prediction + monitoring |
| Prediction logging | SQLite — every prediction logged with timestamp + label |
| Monitoring dashboard | Streamlit page — score trends, label distribution, history |
| CI/CD | GitHub Actions → Render auto-deploy on push to main |
| Containerisation | Dockerfile included |

---

## 👥 Team

IEEE Student Branch — UCSC  
ML Opsidian: Genesis — Final Round
