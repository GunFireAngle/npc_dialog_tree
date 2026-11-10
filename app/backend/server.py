"""NPC 对话树生成器 —— 后端服务。

启动（项目根目录 npc_research/ 下）：
    python -m uvicorn app.backend.server:app --host 127.0.0.1 --port 8000
或：
    python app/backend/server.py

接口：
    GET  /                     前端页面
    GET  /api/worlds           世界观与 NPC 设定
    POST /api/generate         {world_id, depth, model, seed} -> 对话树 + 自动指标
    POST /api/rating           人工盲评量表提交
    GET  /api/ratings/export   导出评分 CSV
    GET  /api/health           模型状态
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from experiment.src.worlds import WORLDS, get_world
from experiment.src.engine import get_engine, MockLLM, LocalLLM
from experiment.src.builder import build_tree
from experiment.src.metrics import compute_metrics

FRONTEND = ROOT / "app" / "frontend"
DATA = ROOT / "app" / "data"
RATING_CSV = DATA / "ratings.csv"
DATA.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NPC 对话树生成器", version="1.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

_lock = threading.Lock()
_engines: dict[str, object] = {}
_load_error: str | None = None


def _engine(kind: str):
    """按需加载并缓存引擎（同一时刻只保留一个，避免 8GB 显存被打爆）。"""
    global _load_error
    if kind in _engines:
        return _engines[kind]
    with _lock:
        if kind in _engines:
            return _engines[kind]
        for k, e in list(_engines.items()):
            if hasattr(e, "unload"):
                e.unload()
            _engines.pop(k, None)
        try:
            eng = get_engine(kind)
            eng.load()
            _engines[kind] = eng
            _load_error = None
        except Exception as e:  # 模型缺失时退回 mock，保证系统仍可演示
            _load_error = f"{type(e).__name__}: {e}"
            eng = MockLLM()
            _engines["mock"] = eng
            return eng
        return _engines[kind]


class GenReq(BaseModel):
    world_id: str = "ash"
    depth: int = 4
    model: str = "base"
    seed: int = 42
    strategy: str = "levelwise"


class RatingReq(BaseModel):
    world_id: str
    depth: int
    model: str
    seed: int
    rater: str = "anon"
    consistency: int
    persona_stability: int
    choice_meaningfulness: int
    comment: str = ""
    blind: bool = True


@app.get("/")
def index():
    return FileResponse(str(FRONTEND / "index.html"))


@app.get("/api/worlds")
def worlds():
    out = []
    for w in WORLDS:
        npc = w["npc"]
        out.append(
            {
                "id": w["id"],
                "name": w["name"],
                "setting": w["setting"],
                "npc": {k: npc[k] for k in ("name", "role", "goal", "style", "values", "facts")},
            }
        )
    return out


@app.get("/api/health")
def health():
    return {
        "engines": list(_engines.keys()),
        "load_error": _load_error,
        "model_path": os.environ.get("NPC_MODEL_PATH", "models/Qwen2.5-1.5B-Instruct"),
    }


@app.post("/api/generate")
async def generate(req: GenReq):
    try:
        world = get_world(req.world_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown world")
    if req.depth not in (2, 3, 4, 5, 6):
        raise HTTPException(status_code=400, detail="depth must be 2..6")
    kind = "mock" if _load_error and req.model != "mock" else req.model

    def work():
        eng = _engine(kind)
        t0 = time.time()
        tree = build_tree(eng, world, world["npc"], req.depth, req.strategy, branch=2, seed=req.seed)
        metrics = compute_metrics(tree, world, world["npc"])
        return {
            "tree": tree.to_dict(),
            "metrics": metrics,
            "engine": getattr(eng, "name", kind),
            "elapsed": round(time.time() - t0, 2),
            "fallback": kind != req.model,
            "load_error": _load_error,
        }

    return await run_in_threadpool(work)


@app.post("/api/rating")
def rating(req: RatingReq):
    new = not RATING_CSV.exists()
    with open(RATING_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "rater", "world_id", "depth", "model", "seed", "blind",
                        "consistency", "persona_stability", "choice_meaningfulness", "comment"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), req.rater, req.world_id, req.depth, req.model,
                    req.seed, int(req.blind), req.consistency, req.persona_stability,
                    req.choice_meaningfulness, req.comment.replace("\n", " ")])
    return {"ok": True, "file": str(RATING_CSV)}


@app.get("/api/ratings/export")
def export_ratings():
    if not RATING_CSV.exists():
        return PlainTextResponse("ts,rater,world_id,depth,model,seed,blind,consistency,persona_stability,choice_meaningfulness,comment\n",
                                 media_type="text/csv")
    return FileResponse(str(RATING_CSV), filename="npctree_ratings.csv", media_type="text/csv")


@app.get("/api/ratings")
def list_ratings():
    if not RATING_CSV.exists():
        return {"rows": []}
    with open(RATING_CSV, encoding="utf-8-sig") as f:
        return {"rows": list(csv.DictReader(f))}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
