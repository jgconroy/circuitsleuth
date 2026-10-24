"""Assemble the self-contained big-machine folder: dist/circuitsleuth_bigmachine (+ zip)."""
import os, shutil, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dst = os.path.join(root, "dist", "circuitsleuth_bigmachine")
if os.path.exists(dst):
    shutil.rmtree(dst)
os.makedirs(dst)
shutil.copytree(os.path.join(root, "circuitsleuth"), os.path.join(dst, "circuitsleuth"), ignore=shutil.ignore_patterns("__pycache__"))
for fn in os.listdir(os.path.join(root, "bigmachine")):
    shutil.copy(os.path.join(root, "bigmachine", fn), dst)
shutil.make_archive(dst, "zip", os.path.dirname(dst), os.path.basename(dst))
print("wrote", dst, "and", dst + ".zip")
