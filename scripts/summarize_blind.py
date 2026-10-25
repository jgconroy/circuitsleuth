"""Summarise blind-benchmark tables (big-machine `blind_<platform>.jsonl` from bigmachine/run_blind.py, or the laptop
`artifacts/blind_*.json` lists from demo.py) into one Markdown table for the README / Devpost.
    python scripts/summarize_blind.py artifacts/lanthanum_blind_circuit.jsonl artifacts/blind_atom.jsonl ...
A row's parameters count as "within 5 %" when max_err < 0.05 (run_blind.py already uses symmetry-aware metrics: NV field vector up to
tetrahedral symmetry, Rydberg Stark curvatures alpha*field_scale^2, offsets modulo the period)."""
import json
import os
import sys

import numpy as np


def load(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path) if l.strip()]
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("rows", [])


def summarise(path):
    rows = load(path)
    name = os.path.basename(path)
    platform = next((p for p in ("circuit", "atom", "ion", "dot", "nv") if p in name), "?")
    n = len(rows)
    ok = [r for r in rows if r.get("family_ok")]
    errs = [r for r in rows if r.get("error")]
    skip = ("fr", "g", "theta", "phi", "x0")  # judged separately (need a crossing / symmetry-equivalent / offset)
    for r in rows:  # laptop tables carry errs but no max_err
        if r.get("max_err") is None and r.get("errs"):
            vals = [v for k, v in r["errs"].items() if k not in skip and isinstance(v, (int, float))]
            r["max_err"] = max(vals) if vals else None
    good = [r for r in ok if r.get("max_err") is not None and r["max_err"] < 0.05]
    rt = [r.get("runtime_s") for r in rows if r.get("runtime_s") is not None]
    by = {}
    for r in rows:
        by.setdefault(str(r.get("truth")), []).append(bool(r.get("family_ok")))
    fam_txt = "; ".join(f"{k} {sum(v)}/{len(v)}" for k, v in sorted(by.items()))
    worst = {}
    for r in ok:
        for k, v in (r.get("errs") or {}).items():
            if isinstance(v, (int, float)) and v >= 0.05:
                worst[k] = worst.get(k, 0) + 1
    worst_txt = ", ".join(f"{k} ({c})" for k, c in sorted(worst.items(), key=lambda kv: -kv[1])[:4]) or "–"
    return dict(platform=platform, file=name, n=n, fam=len(ok), good=len(good), err=len(errs), rt=float(np.mean(rt)) if rt else float("nan"),
                fam_txt=fam_txt, worst_txt=worst_txt)


def main():
    out = [summarise(p) for p in sys.argv[1:]]
    print("| platform | maps | family identified | family + all parameters within 5 % | mean runtime | parameters most often off | source |")
    print("|---|---|---|---|---|---|---|")
    for s in out:
        print(f"| {s['platform']} | {s['n']} | {s['fam']} ({100*s['fam']/max(s['n'],1):.0f} %) | {s['good']} ({100*s['good']/max(s['n'],1):.0f} %) | "
              f"{s['rt']:.0f} s | {s['worst_txt']} | `{s['file']}` |")
    print()
    for s in out:
        print(f"- {s['platform']}: by true family: {s['fam_txt']}" + (f"; {s['err']} runs errored" if s['err'] else ""))


if __name__ == "__main__":
    main()
