"""
routers/train.py
Model Train routes

  POST /train              start training (background)
  GET  /train/status       training progress
  GET  /train/models       model availability + saved file info
  GET  /train/results/all  all model results
  GET  /train/results/{model}  result JSON + image URLs
"""

import json
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter
from fastapi.background import BackgroundTasks
from pydantic import BaseModel

from Models._registry import get_registry_by_section

import config

router = APIRouter(prefix="/train", tags=["Model Train"])

train_status = {"status": "idle", "message": "", "result": None}


class TrainRequest(BaseModel):
    model: str = "static"


def run_training(model_group: str):
    train_status["status"]  = "running"
    train_status["message"] = f"Training {model_group}..."
    train_status["result"]  = None

    try:
        result = subprocess.run(
            ["python", "train.py", "--model", model_group],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])

        from compare import collect_results
        results = collect_results("train")

        train_status["status"]  = "done"
        train_status["message"] = "Training complete"
        train_status["result"]  = results

    except Exception as e:
        train_status["status"]  = "error"
        train_status["message"] = str(e)


@router.post("")
def train(req: TrainRequest, bg: BackgroundTasks):
    if train_status["status"] == "running":
        return {"message": "Training already in progress"}

    train_status["status"]  = "pending"
    train_status["message"] = "Starting..."
    train_status["result"]  = None
    bg.add_task(run_training, req.model)
    return {"message": f"Training started: {req.model}"}


@router.get("/status")
def get_train_status():
    return train_status


@router.get("/models")
def get_models():
    """Return model availability from registry + saved file info.""" 
    models = get_registry_by_section()

    for group in models.values():
        for m in group:
            mid = m["id"]
            model_path  = config.model_save_path(mid)
            result_path = config.result_dir(mid) / "result.json"

            m["has_model"]  = model_path.exists()
            m["model_file"] = model_path.name if model_path.exists() else None
            m["has_result"] = result_path.exists()

            if result_path.exists():
                with open(result_path) as f:
                    m["result"] = json.load(f)
            else:
                m["result"] = None

    return models


@router.get("/results/all")
def get_all_results():
    all_models = [
        "gaussian_process", "xgboost", "random_forest", "mlp",
        "rnn", "lstm", "transformer", "static_time_gnn"
    ]
    results = {}
    for mid in all_models:
        result_path = config.result_dir(mid) / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                results[mid] = json.load(f)
    return results


@router.get("/results/{model_name}")
def get_results(model_name: str):
    result_dir = config.result_dir(model_name)

    train_json = result_dir / "train_result.json"
    result = {}
    if train_json.exists():
        with open(train_json) as f:
            result = json.load(f)

    images = {}
    for img in result_dir.glob("*.png"):
        images[img.stem] = f"/static/{model_name}/{img.name}"

    return {
        "model"  : model_name,
        "result" : result,
        "images" : images,
    }