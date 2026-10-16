"""Compatibility entry point; implementation lives in gwak/sky_position_antenna_patterns.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gwak import sky_position_antenna_patterns as _implementation
for _name in dir(_implementation):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_implementation, _name)
if __name__ == '__main__':
    main()
