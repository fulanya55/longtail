"""Aggregate the V3 formal baseline using the frozen V2 gate calculator."""
from __future__ import annotations
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'experiments/analyze_route_a_frequency_reconstruction_v2.py'
def main():
    text=SRC.read_text(encoding='utf-8')
    for old,new in {
      'route_a_frequency_reconstruction_v2':'route_a_frequency_representation_v3',
      '260821':'261021','260822':'261022','260823':'261023','260824':'261024','260825':'261025',
    }.items(): text=text.replace(old,new)
    ns={'__name__':'__main__','__file__':str(SRC)}
    exec(compile(text,str(SRC)+' [V3 in-memory]','exec'),ns)
if __name__=='__main__': main()
