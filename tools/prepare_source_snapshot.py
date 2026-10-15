"""Copy text sources from the original research projects without changing them.

Run from the repository root. Binary datasets, checkpoints, caches and generated
run directories stay outside Git. Every copied file is recorded with SHA-256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

SKIP_DIRS = {'.git', '.obsidian', '__pycache__', '.pytest_cache', '.cache',
             'outputs', 'generated_data', 'node_modules', '.venv'}
TEXT_SUFFIXES = {'.py', '.m', '.md', '.json', '.yaml', '.yml', '.csv',
                 '.ps1', '.txt', '.toml', '.ini', '.cfg'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=Path('E:/LJQ_code'))
    parser.add_argument('--paper-root', type=Path, default=Path('E:/论文相关/干扰数据'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    generators = Path('干扰数据/干扰信号生成程序')
    sources = [
        (args.code_root / 'CJ_Imbalance_0716', Path('projects/CJ_Imbalance_0716')),
        (args.code_root / 'CJ_Inbalance_0724', Path('projects/CJ_Inbalance_0724')),
        (args.code_root / generators, Path('data_generation/0716')),
        (args.paper_root / '干扰信号生成程序', Path('data_generation/paper')),
    ]
    for source, target in sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
        if (repo / target).exists():
            raise FileExistsError(f'Refusing to overwrite existing snapshot: {target}')
    rows = []
    excluded = Counter()
    for source, target in sources:
        for folder, dirs, files in os.walk(source):
            dirs.sort()
            for directory in dirs:
                if directory in SKIP_DIRS:
                    excluded[directory + '/'] += 1
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                original = Path(folder) / name
                if original.suffix.lower() not in TEXT_SUFFIXES and name != '.gitignore':
                    excluded[original.suffix.lower() or '<no extension>'] += 1
                    continue
                if original.stat().st_size > 10 * 1024 * 1024:
                    raise ValueError(f'Unexpected large text file: {original}')
                relative = target / original.relative_to(source)
                destination = repo / relative
                before = sha256(original)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
                if sha256(destination) != before or sha256(original) != before:
                    raise RuntimeError(f'Source changed while copying: {original}')
                rows.append({'repository_path': relative.as_posix(),
                             'source_path': str(original),
                             'bytes': destination.stat().st_size, 'sha256': before})
    output = repo / 'docs/release'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'source_inventory.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['repository_path', 'source_path', 'bytes', 'sha256'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['repository_path']))
    summary = {
        'snapshot_date': '2026-10-16', 'files': len(rows),
        'bytes': sum(row['bytes'] for row in rows),
        'roots': [{'source': str(s), 'destination': t.as_posix()} for s, t in sources],
        'excluded': dict(sorted(excluded.items())),
        'source_files_modified': False,
        'note': 'Text source snapshot only. Historical run artifacts and datasets remain external.',
    }
    (output / 'source_snapshot.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
