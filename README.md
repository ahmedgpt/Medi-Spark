# 🩺 MediSpark

**Intelligent Symptom Analysis & Healthcare Advisory Platform**

MediSpark is a full-stack healthcare AI application that analyses symptoms, predicts diseases, assesses risk levels, and provides personalised medical guidance — all powered by machine learning, natural language processing, and a real-time streaming pipeline.

> ⚠️ **Disclaimer:** This is an educational project. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Disease Prediction** | Top-3 disease predictions with confidence scores using XGBoost |
| 📊 **Severity Scoring** | 0–100 severity gauge based on symptoms, duration, and age |
| ⚡ **Risk Assessment** | HIGH / MEDIUM / LOW classification with recommended tests |
| 💊 **Medicine Suggestions** | OTC vs prescription recommendations per predicted disease |
| 📚 **RAG Knowledge Base** | Medical Q&A from MedQuAD, WHO fact sheets, and PubMedQA via ChromaDB |
| 💬 **AI Chatbot** | Multi-turn conversational AI with LangChain + Redis-backed memory |
| 🇵🇰 **Roman Urdu Support** | Language detection + 60-entry medical dictionary + Google Translate |
| 📈 **Dashboard** | Chart.js visualisations — severity gauge, risk distribution, disease trends |
| ⚙️ **PySpark Processing** | Batch aggregation of symptom logs every 6 hours into MongoDB trends |
| 🔄 **Continuous Learning** | Auto-retrain + A/B test + deploy when new labelled data accumulates |
| 🔒 **Security** | Rate limiting, input sanitisation, security headers, login-protected APIs |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Flask + Jinja)                │
│   Dashboard │ Predict │ Chat │ History │ Login/Register        │
└────────┬───────────────────────────────────────────────────────┘
         │  REST API
┌────────▼───────────────────────────────────────────────────────┐
│                     FLASK APP (app/__init__.py)                │
│   Routes: predict, chat, dashboard, history, auth             │
│   Security: Flask-Limiter, sanitisation, headers              │
└───┬──────────┬───────────┬────────────┬───────────────────────┘
    │          │           │            │
┌───▼───┐ ┌───▼────┐ ┌────▼─────┐ ┌───▼──────┐
│  ML   │ │  RAG   │ │ Chatbot  │ │  Kafka   │
│Predict│ │ChromaDB│ │LangChain │ │Producer  │
│XGBoost│ │  + ST  │ │+ Redis   │ │          │
└───────┘ └────────┘ └──────────┘ └───┬──────┘
                                      │ Topics
                    ┌─────────────────▼─────────────────────┐
                    │          KAFKA BROKER                  │
                    │  symptom-input │ prediction-result     │
                    │  chat-messages │ audit-log │ retrain   │
                    └──────┬────────────────────────────────┘
                           │
              ┌────────────▼────────────────────┐
              │       KAFKA CONSUMER            │
              │  ML Pipeline │ Retrain Trigger  │
              └──────┬────────────┬─────────────┘
                     │            │
              ┌──────▼──┐  ┌─────▼──────────┐
              │ MongoDB │  │ PySpark Batch   │
              │         │  │ + Continuous    │
              │         │  │   Learner       │
              └─────────┘  └────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop (for Kafka, Redis, MongoDB)

### 1. Clone & install
```bash
git clone https://github.com/your-repo/MediSpark.git
cd MediSpark
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

### 2. Start infrastructure
```bash
docker compose -f docker/docker-compose.yml up -d
```

### 3. Create Kafka topics
```bash
python kafka/create_topics.py
```

### 4. Train the baseline model
```bash
python model_trainer.py
```
A real dataset can be placed at `data/symptom_dataset/dataset.csv`. Without one, a synthetic toy dataset is generated automatically.

### 5. Run the app
```bash
python run.py
```
Open **http://localhost:5000** — register a user, then explore Predict, Chat, and Dashboard.

### 6. Run Kafka consumer (separate terminal)
```bash
python -m app.services.kafka_consumer
```

### 7. Run tests
```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
MediSpark/
├── app/
│   ├── __init__.py              # Flask app factory (security hardened)
│   ├── extensions.py            # MongoDB, Redis, LoginManager
│   ├── routes/
│   │   ├── auth.py              # Registration, login, logout
│   │   ├── predict.py           # POST /api/predict — full ML pipeline
│   │   ├── chat.py              # POST /api/chat — multi-turn chatbot
│   │   ├── dashboard.py         # Dashboard + /api/dashboard/stats
│   │   └── history.py           # GET /api/history
│   ├── services/
│   │   ├── ml_predictor.py      # XGBoost disease prediction
│   │   ├── rag_engine.py        # ChromaDB + sentence-transformers RAG
│   │   ├── severity_scorer.py   # 0–100 severity calculation
│   │   ├── risk_assessor.py     # HIGH/MED/LOW risk + tests + advice
│   │   ├── medicine_suggester.py# OTC/prescription medicine lookup
│   │   ├── chatbot.py           # LangChain chatbot + Redis memory
│   │   ├── urdu_translator.py   # Roman Urdu NLP + deep-translator
│   │   ├── kafka_producer.py    # Kafka event publisher
│   │   └── kafka_consumer.py    # Kafka consumer worker
│   ├── spark/
│   │   ├── batch_processor.py   # PySpark 6h aggregation job
│   │   └── continuous_learner.py# Retrain + A/B test + auto-deploy
│   ├── templates/               # Jinja2 HTML templates
│   └── static/                  # CSS, JS (Chart.js dashboards)
├── kafka/
│   └── create_topics.py         # Kafka topic provisioning
├── docker/
│   ├── Dockerfile               # Production container
│   └── docker-compose.yml       # Full stack (Flask + Kafka + Redis + Mongo)
├── config/settings.py           # Centralised config from .env
├── models/                      # Saved ML models (.pkl)
├── data/                        # Datasets + knowledge base
├── tests/                       # Pytest unit & integration tests
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── run.py                       # Flask entry point
└── PROJECT_PLAN.md              # 4-week development roadmap
```

---

## 🔌 API Reference

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/predict` | Submit symptoms → get disease prediction, risk, medicines | ✅ |
| `POST` | `/api/chat` | Send message → get AI chatbot reply | ✅ |
| `GET` | `/api/chat/history` | Get conversation history | ✅ |
| `DELETE` | `/api/chat/reset` | Clear conversation memory | ✅ |
| `GET` | `/api/history` | Get prediction history | ✅ |
| `GET` | `/api/dashboard/stats` | Get dashboard chart data | ✅ |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Flask, Python 3.11 |
| **ML** | XGBoost, scikit-learn, PySpark MLlib |
| **NLP** | LangChain, sentence-transformers, deep-translator, langdetect |
| **Vector DB** | ChromaDB |
| **Streaming** | Apache Kafka |
| **Database** | MongoDB |
| **Cache/Sessions** | Redis |
| **Frontend** | Jinja2, Chart.js, vanilla CSS/JS |
| **Security** | Flask-Limiter, Flask-Login, bcrypt |
| **Containerisation** | Docker, Docker Compose |

---

## 📅 Development Timeline

| Week | Focus | Status |
|------|-------|--------|
| 1 | Foundation: Flask, Kafka, MongoDB, Redis, baseline ML | ✅ Done |
| 2 | Core AI: RAG, disease prediction, severity, risk, medicines | ✅ Done |
| 3 | Smart features: chatbot, Roman Urdu NLP, PySpark, continuous learning | ✅ Done |
| 4 | Polish: dashboards, Docker, tests, security, documentation | ✅ Done |

---

## 📜 License

This project is for educational purposes. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full development roadmap.
