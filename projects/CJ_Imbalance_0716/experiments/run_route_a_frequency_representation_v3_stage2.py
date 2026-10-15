"""Execute the frozen V2 reconstruction engine with V3 paths/seeds in memory."""
from __future__ import annotations
import runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/run_route_a_frequency_reconstruction_v2.py'

def main() -> None:
    text = SOURCE.read_text(encoding='utf-8')
    for old, new in {
        'dual-axis-frequency-development-v2':'dual-axis-frequency-representation-v3',
        'route_a_frequency_reconstruction_v2':'route_a_frequency_representation_v3',
        '260821':'261021','260822':'261022','260823':'261023','260824':'261024','260825':'261025',
        '3930000001':'3940000001','3930025200':'3940025200',
    }.items():
        text = text.replace(old, new)
    namespace={'__name__':'__main__','__file__':str(SOURCE)}
    # Forward command-line arguments to the reused engine.
    exec(compile(text, str(SOURCE)+' [V3 in-memory]', 'exec'), namespace)

if __name__ == '__main__':
    main()
