import time

import numpy as np

from circuitsleuth import loop, models
from tests.test_platforms import _hidden


def test_autopilot_shrinks_uncertainty_on_nv():
    t, r = _hidden("nv_ensemble", dict(D=2.870, E=0.004, theta=0.6, phi=0.3, field_scale=120.0, x0=0.1), 2.4, 3.35, seed=1)
    t0 = time.time()
    res = loop.run_loop(r["image"], r["x"], r["f"], ("nv_ensemble",), rounds=3, initial_frac=0.35, cfg_kwargs=dict(maxiter=40, workers=1))
    rows = loop.sigma_table(res)
    assert len(rows) >= 2, res["rounds"]
    first, last = rows[0], rows[-1]
    assert last["n_points"] > first["n_points"]
    # the design loop must not make things worse and should tighten the field calibration
    assert last["field_scale_rel_sigma"] <= first["field_scale_rel_sigma"] * 1.05
    assert abs(last["D"] - 2.870) < 0.003
    assert time.time() - t0 < 120
