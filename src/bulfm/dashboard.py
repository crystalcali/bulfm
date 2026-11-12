from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import __version__
from .manager import list_runs
from .storage import safe_run, snapshot


def create_app(root="runs"):
    root = Path(root).resolve()
    app = FastAPI(title="BULFM training dashboard")

    @app.get("/health")
    def health():
        return {"app": "bulfm", "version": __version__, "runs_dir": str(root)}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return files("bulfm").joinpath("static/dashboard.html").read_text()

    @app.get("/api/runs")
    def runs():
        # The list stays small; detailed metrics are loaded only for the selected run.
        return [{k: v for k, v in row.items() if k != "metrics"} for row in list_runs(root)]

    @app.get("/api/runs/{run_id}")
    def detail(run_id: str):
        try:
            return snapshot(safe_run(root, run_id))
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app
