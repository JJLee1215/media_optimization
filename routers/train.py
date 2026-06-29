"""
routers/train.py
Model Train routes — SSE 로그 스트리밍 추가
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

# 로그 버퍼 (SSE로 전송할 줄들)
log_buffer = []
log_subscribers = []   # 연결된 SSE 클라이언트들


def push_log(line: str):
    """로그 한 줄을 버퍼에 추가하고 구독자에게 전송."""
    log_buffer.append(line)
    for q in log_subscribers:
        try:
            q.put_nowait(line)
        except Exception:
            pass


class TrainRequest(BaseModel):
    model       : str  = "static"
    use_pipeline: bool = False


def run_training(model_group: str, use_pipeline: bool = False):
    global log_buffer
    log_buffer = []

    train_status["status"]  = "running"
    train_status["message"] = f"Training {model_group}..."
    train_status["result"]  = None

    push_log(f"═══════════════════════════════════════")
    push_log(f"▶ Training: {model_group.upper()}  |  Pipeline: {'ON' if use_pipeline else 'OFF'}")

    try:
        cmd = ["python", "train.py", "--model", model_group]
        if use_pipeline:
            cmd.append("--pipeline")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            bufsize=1,
        )

        # stdout 한 줄씩 실시간 전송
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
    bg.add_task(run_training, req.model, req.use_pipeline)
    return {"message": f"Training started: {req.model}"}


@router.get("/stream")
async def stream_logs():
    """SSE 엔드포인트 — 로그를 실시간으로 스트리밍."""
    import asyncio
    queue = asyncio.Queue()
    log_subscribers.append(queue)

    # 기존 버퍼 먼저 전송
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