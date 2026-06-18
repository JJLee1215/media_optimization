"""
routers/data.py
Data Analysis routes

  POST /data/upload         upload CSV to data_file/
  GET  /data/columns        read columns + data from CSV → return JSON
  POST /data/analyze        run data_analyzer.py → return results + image URLs
  GET  /data/datasets       list CSV files in data_file/
"""

import shutil
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

import config

router = APIRouter(prefix="/data", tags=["Data Analysis"])


class AnalyzeRequest(BaseModel):
    filename: str


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV to data_file/"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.DATA_DIR / file.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "message" : "Upload successful",
        "filename": file.filename,
        "path"    : str(save_path),
    }


@router.get("/columns")
def get_columns(filename: str, type: str = "static"):
    """
    Read CSV and return:
      - columns list
      - batch_ids (for timeseries)
      - data as JSON (for frontend rendering)

    type: "static" | "timeseries"
    """
    file_path = config.DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    df = pd.read_csv(file_path)

    # Replace NaN with None for JSON serialization
    df = df.where(pd.notnull(df), None)

    if type == "timeseries":
        # Detect batch column
        batch_col = next(
            (c for c in df.columns if "batch" in c.lower()), None
        )
        time_col = next(
            (c for c in df.columns if "time" in c.lower() or "day" in c.lower()), None
        )

        # Skip meta columns
        skip_cols = set()
        if batch_col: skip_cols.add(batch_col)
        if time_col:  skip_cols.add(time_col)
        skip_cols.update([c for c in df.columns if "fault" in c.lower()])
        skip_cols.update([c for c in df.columns if "titer" in c.lower()])

        feature_cols = [c for c in df.columns if c not in skip_cols]
        batch_ids    = sorted(df[batch_col].unique().tolist()) if batch_col else []

        # Build data: {batch_id: {col: [values], time: [times]}}
        data = {}
        if batch_col and time_col:
            for bid, grp in df.groupby(batch_col):
                grp = grp.sort_values(time_col)
                data[str(bid)] = {
                    "time": grp[time_col].tolist(),
                    **{col: grp[col].tolist() for col in feature_cols}
                }

        return {
            "type"       : "timeseries",
            "filename"   : filename,
            "columns"    : feature_cols,
            "batch_col"  : batch_col,
            "time_col"   : time_col,
            "batch_ids"  : batch_ids,
            "n_rows"     : len(df),
            "data"       : data,
        }

    else:
        # Static
        skip_cols = {"batch_id"}
        feature_cols = [c for c in df.columns if c not in skip_cols]

        # Build data: {col: [values]}
        data = {col: df[col].tolist() for col in feature_cols}

        return {
            "type"      : "static",
            "filename"  : filename,
            "columns"   : feature_cols,
            "n_rows"    : len(df),
            "data"      : data,
        }


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run data_analyzer on uploaded file → return summary + image URLs."""
    from data_analyzer import analyze as run_analyze

    file_path = config.DATA_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.filename}")

    out_dir = config.RESULTS_TT_DIR / "analysis"
    summary = run_analyze(str(file_path), str(out_dir))

    summary["plot_urls"] = [
        f"/static/analysis/{Path(p).name}" for p in summary.get("plots", [])
    ]

    return summary


@router.get("/datasets")
def list_datasets():
    """List CSV files in data_file/"""
    files = [f.name for f in config.DATA_DIR.glob("*.csv")]
    return {"datasets": files}