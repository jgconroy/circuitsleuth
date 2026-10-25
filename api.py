"""qĤolmes REST API: the identification engine without the Streamlit UI.

Run:   uvicorn api:app --host 0.0.0.0 --port 8000        (interactive docs at /docs; run from the repo root)
Try:   curl -s localhost:8000/health
       curl -s -X POST localhost:8000/identify/demo -H 'content-type: application/json' \
            -d '{"platform": "nv", "name": "F", "preset": "fast"}'

Endpoints
  GET  /health          liveness and package version
  GET  /demos           demo names per platform (the leading letter is enough for /identify/demo)
  POST /identify        a spectroscopy map {image, x, f} or a point list {points, band}  ->  JSON report
  POST /identify/demo   render one of the built-in hidden devices, identify it, attach the hidden truth

Malformed input is rejected with 422 before any fitting starts; engine failures come back as JSON as well.
NV / dot / ion / atom runs take seconds; superconducting-circuit maps take minutes (global search over three families).
"""
from __future__ import annotations

import json
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

import circuitsleuth
from circuitsleuth import agent, models
from ui.demos import DEMOS, make_demo

PRESETS = {  # search effort -> agent settings (identical to app.py)
    "fast": dict(maxiter=30, n_starts=1, popsize=12, top_k=2, coupled_trials=1),
    "normal": dict(maxiter=50, n_starts=2, popsize=15, top_k=3, coupled_trials=2),
    "thorough": dict(maxiter=100, n_starts=3, popsize=20, top_k=3, coupled_trials=2),
}
Platform = Literal["circuit", "nv", "dot", "ion", "atom"]

app = FastAPI(title="qĤolmes API", version=circuitsleuth.__version__, description=__doc__)


# ------------------------------------------------------------------ request models
class Options(BaseModel):
    """Search settings shared by both identify endpoints."""

    preset: Literal["fast", "normal", "thorough"] = "fast"
    families: list[str] | None = Field(None, description="restrict the candidate families, e.g. ['fluxonium']; default: every family of the platform")
    try_coupled: Literal["auto", "always", "never"] = Field("auto", description="also try a readout-resonator model (circuits only)")
    workers: int = Field(2, ge=1, le=64, description="processes used by the global search")
    use_classifier: bool = Field(True, description="let the CNN proposer rank circuit families and seed the search")

    @field_validator("families")
    @classmethod
    def _known_families(cls, fams):
        if fams is not None:
            bad = [f for f in fams if f not in models.FAMILIES]
            if bad or not fams:
                raise ValueError(f"unknown families {bad}; choose from {sorted(models.FAMILIES)}")
            if len({models.platform_of(f) for f in fams}) > 1:
                raise ValueError("all families must belong to one platform")
        return fams

    def agent_config(self, platform: str = "circuit") -> agent.AgentConfig:
        """Translate the request into the engine's AgentConfig (resonator trials only make sense for circuits)."""
        fams = tuple(self.families or ([] if platform == "circuit" else models.PLATFORMS[platform]["families"]))
        is_circuit = (models.platform_of(fams[0]) if fams else "circuit") == "circuit"
        return agent.AgentConfig(workers=self.workers, families=fams, use_classifier=self.use_classifier,
                                 try_coupled=self.try_coupled if is_circuit else "never", **PRESETS[self.preset])


class IdentifyRequest(Options):
    """Either a map (image[n_f][n_x], x[n_x], f[n_f] in the platform's frequency unit) or a point list
    (rows [x, f] or [x, f, weight]) with its frequency band [f_lo, f_hi]."""

    image: list[list[float]] | None = None
    x: list[float] | None = Field(None, description="control values per column; with points: the full control grid")
    f: list[float] | None = Field(None, description="frequencies per row, strictly increasing")
    points: list[list[float]] | None = None
    band: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_shapes(self):
        if self.band is not None and not self.band[0] < self.band[1]:
            raise ValueError("band must be [f_lo, f_hi] with f_lo < f_hi")
        if self.points is not None:
            if len(self.points) < 10:
                raise ValueError("points: at least 10 rows are needed")
            if any(len(p) not in (2, 3) for p in self.points):
                raise ValueError("points: every row must be [x, f] or [x, f, weight]")
            return self
        if self.image is None or self.x is None or self.f is None:
            raise ValueError("send either image + x + f, or points (+ band)")
        nf, nx, widths = len(self.f), len(self.x), sorted({len(row) for row in self.image})
        if nf < 2 or nx < 2 or len(self.image) != nf or widths != [nx]:
            raise ValueError(f"image must have len(f) = {nf} rows of len(x) = {nx} values; got {len(self.image)} rows of width {widths}")
        if not (np.all(np.diff(self.f) > 0) and np.all(np.isfinite(self.x))):
            raise ValueError("f must be strictly increasing and x finite")
        return self


class DemoRequest(Options):
    platform: Platform
    name: str = Field(..., description="a name from GET /demos, or just its leading letter, e.g. 'F'")
    seed: int = Field(1, ge=0, description="used by the 'Surprise' demos only")


# ------------------------------------------------------------------ helpers
def _jsonable(obj) -> dict:
    """numpy scalars -> floats and NaN/inf -> null, so every client gets strict JSON."""
    return json.loads(agent.to_json(obj), parse_constant=lambda _: None)


def _identify(cfg: agent.AgentConfig, **data) -> dict:
    """Run the engine; unusable input is a 422 with the engine's message, anything else a 500 with the exception."""
    try:
        rep = _jsonable(agent.identify(config=cfg, **data))
    except AssertionError as e:  # the engine's own argument checks
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    if rep.get("status") != "ok":  # e.g. fewer than 10 spectral features
        raise HTTPException(422, rep)
    return rep


# ------------------------------------------------------------------ endpoints (sync functions: FastAPI runs them in a thread pool)
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": circuitsleuth.__version__}


@app.get("/demos")
def demos() -> dict[str, list[str]]:
    """Demo names per platform; /identify/demo accepts the full name or its leading letter."""
    return {platform: list(names) for platform, names in DEMOS.items()}


@app.post("/identify")
def identify(req: IdentifyRequest) -> dict:
    """Identify a device from a map or a point list; returns the full report (best hypothesis, params +- sigma,
    every hypothesis tried, unexplained features, next measurements, experimental design, derived quantities)."""
    cfg = req.agent_config()
    if req.points is not None:
        pts = np.asarray([p if len(p) == 3 else [*p, 1.0] for p in req.points], float)
        return _identify(cfg, points=pts, band=req.band, x=np.asarray(req.x, float) if req.x else None)
    return _identify(cfg, image=np.asarray(req.image, float), x=np.asarray(req.x, float), f=np.asarray(req.f, float), band=req.band)


@app.post("/identify/demo")
def identify_demo(req: DemoRequest) -> dict:
    """Render a built-in hidden device deterministically and identify it; the report gains a `truth` key."""
    names = DEMOS[req.platform]
    name = next((n for n in names if n == req.name or n.split(" ·")[0] == req.name), None)
    if name is None:
        raise HTTPException(404, f"no demo {req.name!r} on platform {req.platform!r}; see GET /demos")
    truth, r = make_demo(req.platform, name, req.seed)
    rep = _identify(req.agent_config(req.platform), image=r["image"], x=r["x"], f=r["f"])
    rep["truth"] = _jsonable(truth)
    return rep
