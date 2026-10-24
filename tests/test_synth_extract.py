import numpy as np

from circuitsleuth import extract, models, synth


def test_sample_truth_is_deterministic_and_in_bounds():
    a = synth.sample_truth(np.random.default_rng(3))
    b = synth.sample_truth(np.random.default_rng(3))
    assert a == b
    bounds = models.hypothesis_bounds(a["family"], a["coupled"])
    for k, (lo, hi) in bounds.items():
        assert lo <= a["params"][k] <= hi


def test_render_shapes_and_labels():
    cfg = synth.SynthConfig(nx=61, nf=200)
    rng = np.random.default_rng(1)
    t = synth.sample_truth(rng, cfg, family="fluxonium", coupled=False)
    r = synth.render(t, cfg, rng)
    assert r["image"].shape == (200, 61)
    assert r["points"].shape[1] == 3
    assert len(r["labels"]) == len(r["points"])
    assert np.isfinite(r["image"]).all()
    thumb = synth.downsample(r["image"], 64)
    assert thumb.shape == (64, 64) and thumb.min() >= 0 and thumb.max() <= 1


def test_every_family_renders_with_and_without_resonator():
    cfg = synth.SynthConfig(nx=41, nf=150)
    rng = np.random.default_rng(7)
    for fam in synth.FAMILY_NAMES:
        for coupled in (False, True):
            t = synth.sample_truth(rng, cfg, family=fam, coupled=coupled)
            r = synth.render(t, cfg, rng)
            assert r["image"].shape == (150, 41)


def test_extract_recovers_most_truth_points_on_clean_maps():
    cfg = synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.08), max_spurs=0, linewidth_range=(0.02, 0.05))
    recalls = []
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        t = synth.sample_truth(rng, cfg)
        r = synth.render(t, cfg, rng)
        pts = extract.extract_points(r["image"], r["x"], r["f"])
        # only truth points bright enough to be detectable (>= 5 sigma above the noise)
        truth = r["points"][r["points"][:, 2] >= 5 * t["noise"]]
        recalls.append(extract.match_points(pts, truth, tol_f=1.5 * t["linewidth"]))
    assert np.mean(recalls) >= 0.9, recalls


def test_extract_handles_dips_as_well_as_peaks():
    cfg = synth.SynthConfig(nx=41, nf=300, max_spurs=0, noise_range=(0.05, 0.05))
    rng = np.random.default_rng(5)
    t = synth.sample_truth(rng, cfg, family="tunable_transmon", coupled=False)
    t["sign"] = -1
    r = synth.render(t, cfg, rng)
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    truth = r["points"][r["points"][:, 2] >= 5 * t["noise"]]
    assert extract.match_points(pts, truth, tol_f=1.5 * t["linewidth"]) >= 0.85


def test_extract_precision_on_clean_maps():
    """Most extracted weight must sit on real features, not noise."""
    cfg = synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.1), max_spurs=0)
    precs = []
    for seed in range(6):
        rng = np.random.default_rng(200 + seed)
        t = synth.sample_truth(rng, cfg)
        r = synth.render(t, cfg, rng)
        pts = extract.extract_points(r["image"], r["x"], r["f"])
        tp = r["points"]
        good = 0.0
        for x, f, w in pts:
            same = tp[np.isclose(tp[:, 0], x)]
            if same.size and np.min(np.abs(same[:, 1] - f)) < 2 * t["linewidth"] + 0.02:
                good += w
        precs.append(good / pts[:, 2].sum())
    assert np.mean(precs) > 0.9, precs


def test_tracks_and_flat_detection():
    cfg = synth.SynthConfig(nx=81, nf=400, noise_range=(0.05, 0.05), max_spurs=0)
    rng = np.random.default_rng(9)
    t = synth.sample_truth(rng, cfg, family="tunable_transmon", coupled=False)
    t["spurs"] = [dict(f=0.5 * (t["f_lo"] + t["f_hi"]), amp=0.9, sideband=0.0)]
    r = synth.render(t, cfg, rng)
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    ids = extract.link_tracks(pts)
    flat = extract.flat_track_mask(pts, ids)
    # the spur must be detected as flat; the qubit line must not
    spur_pts = np.abs(pts[:, 1] - t["spurs"][0]["f"]) < 0.03
    assert flat[spur_pts].mean() > 0.8
    q = np.abs(pts[:, 1] - np.interp(pts[:, 0], r["x"], r["lines"]["0-1"])) < 0.03
    assert flat[q].mean() < 0.2
    summ = extract.summarize_tracks(pts, ids, flat)
    assert any(abs(s["f_mean"] - t["spurs"][0]["f"]) < 0.03 for s in summ)
