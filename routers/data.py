"""
routers/data.py
Data routes

  POST /data/upload              CSV 파일 업로드
  GET  /data/datasets            업로드된 파일 목록
  GET  /data/columns             컬럼 목록 + 배치 목록
  POST /data/analyze             시계열 배치 데이터 JSON 반환 (Timepanel용)
  GET  /data/analyze/static      static 분석 (PNG or JSON)
  GET  /data/analyze/timeseries  timeseries 분석 (PNG)
"""

import json
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import config
from data_analyzer import (
    basic_stats,
    missing_analysis,
    correlation_heatmap,
    correlation_stats,
    distribution_plots,
    outlier_plots,
    titer_correlation_plots,
    pca_plots,
    timeseries_profile,
)

router = APIRouter(prefix="/data", tags=["Data"])

UPLOAD_DIR = config.DATA_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Upload ────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """CSV 파일 업로드 → data_file/ 저장."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    dest    = UPLOAD_DIR / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    return {"filename": file.filename, "path": str(dest), "size": len(content)}


# ── Datasets ─────────────────────────────────────────────────────────────

@router.get("/datasets")
def list_datasets():
    """업로드된 CSV 파일 목록 반환."""
    files = list(UPLOAD_DIR.glob("*.csv"))
    return [{"filename": f.name, "size": f.stat().st_size} for f in files]


# ── Columns ──────────────────────────────────────────────────────────────

@router.get("/columns")
def get_columns(filename: str, type: str = "timeseries"):
    """
    CSV 파일의 컬럼 목록과 배치 목록 반환.

    type = "static"     : 배지 컴포넌트 컬럼 + 배치 목록
    type = "timeseries" : 시계열 컬럼 + 배치 목록
    """
    file_path = config.DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)

        skip_cols = ["Batch_ID", "Time (day)", "Fault flag"]
        columns   = [c for c in df.columns if c not in skip_cols]

        batches = []
        if "Batch_ID" in df.columns:
            batches = sorted(df["Batch_ID"].unique().tolist())

        return {
            "filename" : filename,
            "type"     : type,
            "columns"  : columns,
            "batches"  : batches,
            "rows"     : len(df),
            "n_batches": len(batches) if batches else len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Analyze POST (Timepanel용 — 배치별 시계열 JSON) ──────────────────────

class AnalyzeRequest(BaseModel):
    filename : str
    type     : str  = "timeseries"
    columns  : list = []
    batch_id : str  = "all"


@router.post("/analyze")
def analyze_post(req: AnalyzeRequest):
    """
    POST /data/analyze — Timepanel에서 호출하는 배치별 시계열 데이터 반환.

    Returns JSON:
    {
      "1": { "time": [1,2,...14], "Glucose_conc": [...], ... },
      "2": { ... },
      ...
    }
    """
    file_path = config.DATA_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)

        batch_col = "Batch_ID"
        time_col  = "Time (day)"
        skip_cols = [batch_col, time_col, "Fault flag"]

        # 선택된 컬럼 or 전체
        feat_cols = req.columns if req.columns else [
            c for c in df.columns if c not in skip_cols
        ]

        # Fault flag 필터
        if "Fault flag" in df.columns:
            df = df[df["Fault flag"] == 0]

        # 배치 필터
        if req.batch_id and req.batch_id != "all":
            df = df[df[batch_col] == int(req.batch_id)]

        result = {}
        for bid, grp in df.groupby(batch_col):
            grp   = grp.sort_values(time_col)
            entry = {"time": grp[time_col].tolist()}
            for col in feat_cols:
                if col in grp.columns:
                    entry[col] = grp[col].tolist()
            result[str(int(bid))] = entry

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Static analysis (GET) ─────────────────────────────────────────────────

@router.get("/analyze/static")
def analyze_static(
    type         : str = "stats",
    filepath     : str = None,
    batch_id     : str = None,
    selected_cols: str = None,
):
    """
    Static data 분석.

    type:
      stats             → JSON  기초 통계
      missing           → JSON  결측치
      heatmap           → PNG   컴포넌트 간 상관관계 히트맵
      correlation_stats → PNG   Pearson/Spearman/p-value vs Titer
      distribution      → PNG   히스토그램
      outlier           → PNG   IQR boxplot
      titer             → PNG   컴포넌트 vs Titer 산점도
      pca               → PNG   PCA biplot + 설명분산
    """
    file_path = (config.DATA_DIR / filepath) if filepath else config.DATA_STATIC
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    cols = [c.strip() for c in selected_cols.split(",")] if selected_cols else None

    try:
        if type == "stats":
            return JSONResponse(basic_stats(str(file_path), batch_id=batch_id))

        if type == "missing":
            return JSONResponse(missing_analysis(str(file_path)))

        if type == "heatmap":
            out = correlation_heatmap(str(file_path), cols, batch_id)
        elif type == "correlation_stats":
            out = correlation_stats(str(file_path), cols)
        elif type == "distribution":
            out = distribution_plots(str(file_path), cols, batch_id)
        elif type == "outlier":
            out = outlier_plots(str(file_path), cols)
        elif type == "titer":
            out = titer_correlation_plots(str(file_path), cols)
        elif type == "pca":
            out = pca_plots(str(file_path), cols)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown type: {type}")

        if not Path(out).exists():
            raise HTTPException(status_code=500, detail="Plot generation failed.")
        return FileResponse(out, media_type="image/png")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Timeseries analysis (GET) ─────────────────────────────────────────────

@router.get("/analyze/timeseries")
def analyze_timeseries(
    type    : str = "profile",
    filepath: str = None,
):
    """
    Timeseries data 분석 PNG.

    type:
      profile → PNG  14일 평균 프로파일
    """
    file_path = (config.DATA_DIR / filepath) if filepath else config.DATA_TIMESERIES
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        if type == "profile":
            out = timeseries_profile(str(file_path))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown type: {type}")

        if not Path(out).exists():
            raise HTTPException(status_code=500, detail="Plot generation failed.")
        return FileResponse(out, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))