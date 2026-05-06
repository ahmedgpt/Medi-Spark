# MediSpark — Week 1

Intelligent Symptom Analysis & Healthcare Advisory Platform.
This branch covers the **Week 1 Foundation & Setup** deliverables.

## What's working in Week 1
- ✅ Flask app factory with Redis-backed sessions
- ✅ User registration / login (Flask-Login + MongoDB)
- ✅ Symptom submission UI + REST endpoints (`/api/predict`, `/api/chat`, `/api/history`)
- ✅ Kafka producer (publishes to `symptom-input`) + standalone consumer worker
- ✅ All 5 Kafka topics provisioned via `kafka/create_topics.py`
- ✅ Baseline disease classifier (Random Forest) with synthetic-fallback trainer
- ✅ Docker Compose for Kafka, Zookeeper, Redis, MongoDB
- ✅ Pytest smoke test for severity scorer

Week 2-4 service stubs already exist so future wiring is just an implementation swap.

---

## 1. Prerequisites
- Python 3.10+
- Docker Desktop (for Kafka/Redis/Mongo)

## 2. Clone & install
```powershell
cd f:\MediSpark
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## 3. Start infrastructure
```powershell
docker compose -f docker/docker-compose.yml up -d
```

## 4. Create Kafka topics
```powershell
python kafka/create_topics.py
```

## 5. Train the baseline model
A real Kaggle dataset can be dropped at `data/symptom_dataset/dataset.csv`.
Without one, a synthetic toy dataset is used so the pipeline still runs.
```powershell
python -m app.spark.model_trainer
```

## 6. Run the app
```powershell
python run.py
```
Open http://localhost:5000 — register a user, then go to **Predict**.

## 7. Run the Kafka consumer (separate terminal)
```powershell
python -m app.services.kafka_consumer
```
Submitting symptoms in the UI will appear here as consumed events and be mirrored to `audit-log`.

## 8. Tests
```powershell
pytest -q
```

---

## Project layout
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full 4-week roadmap and folder map.

## Disclaimer
Educational project — not a substitute for professional medical advice.
