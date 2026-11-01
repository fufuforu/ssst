"""Unchanged upstream modules via an isolated, pinned scripts search path."""
from pathlib import Path
import scripts
_path = str(Path(__file__).resolve().parents[1]/'third_party/native_querysplat/scripts')
if _path not in scripts.__path__: scripts.__path__.append(_path)
