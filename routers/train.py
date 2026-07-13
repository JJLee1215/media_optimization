"""
routers/train.py
Model Train routes — SSE 로그 스트리밍 + 파일경로 파라미터화
"""

import json
import os
import subprocess
import asyncio
from pathlib import Path

from fastapi import APIRouter
from fastapi.background import BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from Models._registry import get_registry_by_section
import config

router = APIRouter(prefix="/train", tags=["Model Train"])

train_status = {
    "status"  : "idle",
    "message" : "",
    "result"  : None,
}

log_buffer = []
log_subscribers = []


def push_log(line: str):
    log_buffer.append(line)
    for q in log_subscribers:
        try:
            q.put_nowait(line)
        except Exception:
            pass


class TrainRequest(BaseModel):
    model             : str  = "static"
    use_pipeline      : bool = False
    static_file       : str  = None
    ts_file           : str  = None
    selected_cols     : list[str] | None = None
    selected_ts_cols  : list[str] | None = None


def run_training(model_group: str, use_pipeline: bool = False,
                 static_file: str = None, ts_file: str = None,
                 selected_cols: list = None, selected_ts_cols: list = None):
    global log_buffer
    log_buffer = []

    train_status["status"]  = "running"
    train_status["message"] = f"Training {model_group}..."
    train_status["result"]  = None

    push_log(f"═══════════════════════════════════════")
    push_log(f"▶ Training: {model_group.upper()}  |  Pipeline: {'ON' if use_pipeline else 'OFF'}")
    push_log(f"  Static file : {static_file or config.DATA_STATIC}")
    push_log(f"  TS file     : {ts_file or config.DATA_TIMESERIES}")
    if selected_cols:
        push_log(f"  Selected static cols : {selected_cols}")
    if selected_ts_cols:
        push_log(f"  Selected ts cols     : {selected_ts_cols}")

    try:
        cmd = ["python", "train.py", "--model", model_group]
        if use_pipeline:
            cmd.append("--pipeline")
        if static_file:
            cmd += ["--static_file", static_file]
        if ts_file:
            cmd += ["--ts_file", ts_file]
        if selected_cols:
            cmd += ["--selected_cols", ",".join(selected_cols)]
        if selected_ts_cols:
            cmd += ["--selected_ts_cols", ",".join(selected_ts_cols)]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                push_log(line)

        proc.wait()

        if proc.returncode != 0:
            push_log(f"❌ Error: process exited with code {proc.returncode}")
            raise RuntimeError(f"Training failed (code {proc.returncode})")

        from compare import collect_results
        results = collect_results("train")

        push_log("═══════════════════════════════════════")
        push_log(f"✅ Training complete")

        train_status["status"]  = "done"
        train_status["message"] = "Training complete"
        train_status["result"]  = results

    except Exception as e:
        push_log(f"❌ {str(e)}")
        train_status["status"]  = "error"
        train_status["message"] = str(e)


@router.post("")
def train(req: TrainRequest, bg: BackgroundTasks):
    if train_status["status"] == "running":
        return {"message": "Training already in progress"}

    train_status["status"]  = "pending"
    train_status["message"] = "Starting..."
    train_status["result"]  = None
    bg.add_task(
        run_training,
        req.model,
        req.use_pipeline,
        req.static_file,
        req.ts_file,
        req.selected_cols,
        req.selected_ts_cols,
    )
    return {"message": f"Training started: {req.model}"}


@router.get("/stream")
async def stream_logs():
    queue = asyncio.Queue()
    log_subscribers.append(queue)

    for line in log_buffer:
        await queue.put(line)

    async def event_gen():
        try:
            while True:
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps({'log': line})}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'ping': True})}\n\n"
                    if train_status["status"] in ("done", "error", "idle"):
                        break
        finally:
            log_subscribers.remove(queue)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"              : "no-cache",
            "X-Accel-Buffering"          : "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/status")
def get_train_status():
    return train_status


@router.get("/models")
def get_models():
    models = get_registry_by_section()
    for group in models.values():
        for m in group:
            mid         = m["id"]
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
    result_dir  = config.result_dir(model_name)
    train_json  = result_dir / "result.json"
    result = {}
    if train_json.exists():
        with open(train_json) as f:
            result = json.load(f)
    images = {}
    for img in result_dir.glob("*.png"):
        images[img.stem] = f"/static/{model_name}/{img.name}"
    return {"model": model_name, "result": result, "images": images}


_pipeline_dim_cache = {}

@router.get("/pipeline_dims")
def get_pipeline_dims():
    """
    각 파이프라인의 차원 구성을 실제 모델에서 계산해서 반환.
    embedding/metal_physchem/log_conc/gem 세부 항목 + total을 함께 내려줘서
    프론트가 Heterogeneity 카드 하단 상세 표시와 우측 요약(total)에
    모두 재사용할 수 있게 함. 한 번 계산한 값은 프로세스가 살아있는 동안 캐싱
    (특히 UniMol처럼 무거운 모델은 매 요청마다 다시 로딩하면 느리므로).
    """
    global _pipeline_dim_cache
    if _pipeline_dim_cache:
        return _pipeline_dim_cache

    dims = {}

    # RDKit
    from heterogeneity.smile_gem_pipe import (
        MediaPipeline, METAL_PHYSCHEM_DIM, GEM_DIM
    )
    rdkit_pipe = MediaPipeline()
    dims["rdkit"] = {
        "embedding"      : rdkit_pipe._emb_dim,
        "metal_physchem" : METAL_PHYSCHEM_DIM,
        "log_conc"       : 1,
        "gem"            : GEM_DIM,
        "total"          : rdkit_pipe.vector_dim,
    }

    # ChemBERTa — config만 불러오면 가벼움 (가중치 전체 다운로드 X)
    from transformers import AutoConfig
    from heterogeneity.smile_BERTA_gem_pipe import (
        METAL_PHYSCHEM_DIM as BERTA_METAL_DIM,
        GEM_DIM as BERTA_GEM_DIM,
        CHEMBERTA_MODEL_NAME,
    )
    cfg = AutoConfig.from_pretrained(CHEMBERTA_MODEL_NAME)
    emb_dim = cfg.hidden_size
    dims["chemberta"] = {
        "embedding"      : emb_dim,
        "metal_physchem" : BERTA_METAL_DIM,
        "log_conc"       : 1,
        "gem"            : BERTA_GEM_DIM,
        "total"          : emb_dim + BERTA_METAL_DIM + 1 + BERTA_GEM_DIM,
    }

    # UniMol — 무거움, 여기서만 실제 로딩
    from heterogeneity.smile_UniMol_gem_pipe import (
        UniMolMediaPipeline,
        METAL_PHYSCHEM_DIM as UNIMOL_METAL_DIM,
        GEM_DIM as UNIMOL_GEM_DIM,
    )
    unimol_pipe = UniMolMediaPipeline()
    dims["unimol"] = {
        "embedding"      : unimol_pipe._emb_dim,
        "metal_physchem" : UNIMOL_METAL_DIM,
        "log_conc"       : 1,
        "gem"            : UNIMOL_GEM_DIM,
        "total"          : unimol_pipe.vector_dim,
    }

    _pipeline_dim_cache = dims
    return dims