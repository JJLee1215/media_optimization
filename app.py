"""
app.py
FastAPI entry point

Mounts:
  /static   → Results_Train_Test/ (serve plot images)

Routers:
  /data     → routers/data.py     (upload, analyze, datasets)
  /train    → routers/train.py    (train, status, results)
  /compare  → routers/compare.py  (comparison chart)
  /predict  → routers/predict.py  (titer/viability prediction)
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from routers import data, train, compare, predict

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Bioprocess ML Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Serve result images
config.RESULTS_TT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(config.RESULTS_TT_DIR)), name="static")

# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(data.router)
app.include_router(train.router)
app.include_router(compare.router)
app.include_router(predict.router)


# ── Root ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Bioprocess ML Tool API",
        "routes" : [
            "POST /data/upload",
            "POST /data/analyze",
            "GET  /data/datasets",
            "POST /train",
            "GET  /train/status",
            "GET  /train/results/{model}",
            "GET  /compare",
            "POST /predict",
        ]
    }