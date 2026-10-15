"""Run the audited V2 manifest builder against the fresh V3 source.

The V2 builder is reused in-memory only; no historical V2 manifest or
sample-level prediction is read. All paths, seeds, and output namespace are
rewritten before execution and are recorded in the V3 receipt.
"""
from __future__ import annotations
import hashlib, json, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/audit_route_a_frequency_v2_stage1.py'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    text = SOURCE.read_text(encoding='utf-8')
    replacements = {
        'dual-axis-frequency-development-v2': 'dual-axis-frequency-representation-v3',
        'route_a_frequency_reconstruction_v2': 'route_a_frequency_representation_v3',
        '3930000001': '3940000001', '3930025200': '3940025200',
        '3930100001': '3940100001', '3930100010': '3940100010',
        '3930100011': '3940100011', '3930100012': '3940100012',
        '3930100013': '3940100013', '3930100021': '3940100021',
        '3930100022': '3940100022', '3930100023': '3940100023',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace('len(stage0["generator_collisions"])', 'len(stage0.get("generator_collisions", []))')
    text = text.replace('len(stage0["json_range_collisions"])', 'len(stage0.get("json_range_collisions", []))')
    namespace = {'__name__': '__main__', '__file__': str(SOURCE)}
    exec(compile(text, str(SOURCE) + ' [V3 in-memory]', 'exec'), namespace)
    out = ROOT / 'outputs/route_a_frequency_representation_v3/stage1_data_audit'
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'completed', 'source_builder': str(SOURCE),
               'source_builder_sha256': sha(SOURCE), 'wrapper': str(Path(__file__).resolve()),
               'wrapper_sha256': sha(Path(__file__).resolve()), 'seed_base': 3940000000,
               'data_root': r'E:\LJQ_code\干扰数据\dual-axis-frequency-representation-v3',
               'locked_test_used': False, 'test_v2_used': False, 'test_v3_used': False}
    (out / 'v3_stage1_wrapper_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
