# MediSpark — Phase-by-Phase Development Plan

Intelligent Symptom Analysis & Healthcare Advisory Platform.
Stack: Flask · Kafka · PySpark · LangChain · FAISS/Chroma · XGBoost · MongoDB · Redis · Docker.

---

## Week 1 — Foundation & Setup (Days 1–7) ← **CURRENT**
**Goal:** Environment ready, basic Flask app running, Kafka integrated, baseline ML model trained.

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–2 | Install Python 3.10+, Docker, Kafka, MongoDB, Redis. Create venv. | `requirements.txt`, `docker/docker-compose.yml`, `.env.example` |
| 3–4 | Build Flask base app: registration/login (Flask-Login), symptom form UI, REST skeleton (`/predict`, `/chat`, `/history`), Redis sessions. | `app/__init__.py`, `app/routes/*`, `app/templates/*` |
| 5–6 | Kafka integration: topics (`symptom-input`, `prediction-result`, `audit-log`, `chat-messages`, `alert-high-risk`), `kafka_producer.py`, `kafka_consumer.py`, end-to-end test. | `app/services/kafka_*.py`, `kafka/create_topics.py` |
| 7 | Download symptom-disease dataset, clean with Pandas, train RF/XGBoost, save with joblib. | `app/spark/model_trainer.py`, `models/disease_classifier.pkl` |

**Exit criteria:** `python run.py` serves Flask UI · `docker compose up` starts Kafka/Redis/Mongo · sending a symptom from the form publishes to `symptom-input` and a consumer logs it · trained `.pkl` exists.

---

## Week 2 — Core AI Engine (Days 8–14)
- **Days 8–9:** RAG engine — chunk medical PDFs, embed with sentence-transformers, store in FAISS/Chroma, build retrieval chain (`app/services/rag_engine.py`).
- **Days 10–11:** Disease prediction pipeline — top-3 diseases + confidence + severity 0–100 (`app/services/ml_predictor.py`, `severity_scorer.py`).
- **Days 12–13:** Risk assessor (HIGH/MED/LOW) + diagnostic test suggester (`app/services/risk_assessor.py`).
- **Day 14:** Medicine suggester — vector lookup + LLM dosage generation, OTC vs prescription (`app/services/medicine_suggester.py`).

## Week 3 — Smart Features + PySpark (Days 15–21)
- **Days 15–16:** Multi-turn chatbot with LangChain `ConversationBufferMemory`; stream messages via Kafka.
- **Days 17–18:** Roman Urdu / English NLP — language detection + `deep-translator` + custom medical dictionary (`app/services/urdu_translator.py`).
- **Day 19:** PySpark batch job — aggregate symptom logs every 6h into MongoDB trends (`app/spark/batch_processor.py`).
- **Days 20–21:** Continuous learning — collect new data, retrain via PySpark, A/B test, auto-deploy.

## Week 4 — Polish, Integration & Deployment (Days 22–28)
- **Days 22–23:** UI/UX polish — Chart.js dashboards, severity gauge, risk badge, print/download.
- **Days 24–25:** Docker Compose with all services (flask-app, kafka, zookeeper, pyspark, mongodb, redis, chromadb).
- **Day 26:** Tests — unit/integration + Locust load test (100 concurrent).
- **Day 27:** Security — input sanitization, Flask-Limiter, JWT refresh, disclaimers.
- **Day 28:** Demo prep — video, README, slides, deploy (Railway/Render/AWS EC2).

---

## Repository Layout
```
MediSpark/
├── app/
│   ├── __init__.py            # Flask app factory
│   ├── extensions.py          # db, login_manager, redis client
│   ├── routes/                # auth, predict, chat, history, dashboard
│   ├── models/                # user, symptom_log
│   ├── services/              # kafka, rag, ml, severity, risk, medicine, urdu
│   ├── spark/                 # batch_processor, model_trainer
│   ├── templates/             # Jinja HTML
│   └── static/                # css, js
├── kafka/
│   ├── topics_config.yaml
│   └── create_topics.py
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
├── config/
│   └── settings.py
├── data/
│   ├── medical_knowledge/
│   ├── drug_database/
│   └── symptom_dataset/
├── models/                    # saved .pkl
├── tests/
├── .env.example
├── requirements.txt
├── run.py
└── README.md
```
