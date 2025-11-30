# ProfessionAI_Online_Reputation_Monitoring_Sentiment_Analysis
ProfessionAI Master AI Engineering. Progetto per il corso "MLOps e Machine Learning in Produzione"


# MachineInnovators – Online Reputation Monitoring  
### Sentiment Analysis • FastAPI • Docker • Grafana (Infinity Plugin) • MLOps Pipeline

This project implements a complete system for **online reputation monitoring** through automated **sentiment analysis** of social media text.  
It uses **RoBERTa**, **FastAPI**, **Docker**, and **Grafana Infinity datasource** for Grafana monitoring.

The system is designed as an **MLOps-ready pipeline**, supporting:
- Sentiment model inference via API  
- Containerized deployment  
- Real-time monitoring dashboards  
- Simple retraining workflow  
- CI/CD integration with GitHub Actions  

---

# 1 - Project Objective

Companies must track social media sentiment to maintain a positive online reputation.  
Manual monitoring is slow, expensive, and unreliable.

This project enables:

- Automated classification of user sentiment  
- Continuous monitoring via Grafana dashboards  
- FastAPI-based inference API for easy integration  
- A retraining-ready architecture  
- Containerized deployment for scalability  
- Grafana Monitoring.

---

# 2 - System Architecture
            ┌──────────────────────┐
            │  Social Media Input  │
            └───────────┬──────────┘
                        │ (texts)
                        ▼
              ┌───────────────────┐
              │ FastAPI Inference │  ← Docker container
              └───────────┬───────┘
                 /predict │
                 /stats   ▼
         ┌───────────────────────────┐
         │ Grafana (Infinity Plugin) │  ← pulls JSON from /stats
         │  → Dashboards & Alerts    │
         └───────────────────────────┘

**Grafana Infinity datasource** directly queries the `/stats` endpoint of FastAPI.  

---

# 3 - Sentiment Model

The project uses a Twitter-optimized RoBERTa model: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

Model characteristics:
- Pretrained on millions of tweets
- Optimized for short, informal social media texts
- Outputs: **positive**, **neutral**, **negative**

---

# 4 - Project Structure

```text
.
├── src
│   ├── app.py
│   ├── model.py
│   ├── predict.py
│   ├── monitoring.py
│   ├── train.py
│   └── data.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── tests/
│
├── .github
    └── workflows
        ├── ci.yml                # CI pipeline (tests + linting)
        ├── docker-publish.yml    # Build & publish Docker image
        └── deploy_hf.yml         # Optional: deploy model to HuggingFace
