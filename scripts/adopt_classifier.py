"""Adopt an overnight-trained proposer if it beats the current one.
    python scripts/adopt_classifier.py out/lanthanum_v7/classifier_best.pt [out/qec1_v7/classifier_best.pt ...]
Compares held-out family accuracy (classifier_best.json next to each .pt) with artifacts/classifier.json, copies the best into
artifacts/ (keeping the old files as classifier_prev.*), and checks the new net loads and proposes on a demo map."""
import json
import os
import shutil
import sys

sys.path.insert(0, ".")


def metrics(pt_path):
    js = os.path.splitext(pt_path)[0] + ".json"
    if not os.path.exists(js):
        return None
    m = json.load(open(js))
    return dict(path=pt_path, fam_acc=float(m.get("fam_acc", 0)), coupled_acc=float(m.get("coupled_acc", 0)))


def main():
    cur = json.load(open("artifacts/classifier.json"))
    cur_key = (float(cur["fam_acc"]), float(cur.get("coupled_acc", 0)))
    print(f"current: fam_acc {cur_key[0]:.4f} crossing_acc {cur_key[1]:.4f}")
    cands = [m for m in (metrics(p) for p in sys.argv[1:]) if m]
    for m in cands:
        print(f"candidate {m['path']}: fam_acc {m['fam_acc']:.4f} crossing_acc {m['coupled_acc']:.4f}")
    if not cands:
        print("no candidates with a .json next to them"); return
    best = max(cands, key=lambda m: (m["fam_acc"], m["coupled_acc"]))
    if (best["fam_acc"], best["coupled_acc"]) <= cur_key:
        print("keeping the current proposer (no candidate is better)"); return
    from circuitsleuth import classify
    net = classify.load(best["path"])
    from ui.demos import DEMOS, make_demo
    t, r = make_demo("circuit", list(DEMOS["circuit"])[0], 1)
    props = classify.propose(net, r["image"], r["x"], top_k=3)
    print("loads and proposes:", [(p["family"], round(p["prob"], 2)) for p in props])
    for ext in (".pt", ".json"):
        shutil.copy(f"artifacts/classifier{ext}", f"artifacts/classifier_prev{ext}")
        shutil.copy(os.path.splitext(best["path"])[0] + ext, f"artifacts/classifier{ext}")
    print(f"adopted {best['path']} -> artifacts/classifier.pt (+json); previous kept as classifier_prev.*")
    print("now run: python scripts/apptest_quick.py  and restart streamlit")


if __name__ == "__main__":
    main()
