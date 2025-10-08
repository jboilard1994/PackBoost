# tests/conftest.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if you use src layout, replace ROOT with os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
