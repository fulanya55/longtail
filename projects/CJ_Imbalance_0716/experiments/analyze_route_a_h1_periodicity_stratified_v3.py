"""Apply the frozen F3 stratification protocol to H1/B2/B3 predictions."""
import argparse
from pathlib import Path
import analyze_route_a_isrj_f3_stratified_robustness_v3 as base

PROJECT_ROOT=Path(__file__).resolve().parents[1]

def main():
    base.VARIANTS={"F0":"B0_H1","F3":"B2_H1AlignedPeriodicity","F4":"B3_H1BlockShuffled"}
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/student_a/six_class_development_v3.yaml')
    p.add_argument('--fusion-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_h1_periodicity_increment')
    p.add_argument('--c6-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_development_v3_label_selective_projection')
    p.add_argument('--test-v3-manifest',type=Path,default=PROJECT_ROOT/'frozen/20260731_route_a_test_v3_sealed/test_v3_sealed_manifest.json')
    p.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/tapc/route_a_h1_periodicity_stratified_v3')
    p.add_argument('--seeds',type=int,nargs='+',default=[2039,2040,2041,2042,2043])
    p.add_argument('--weak-threshold-db',type=float,default=10.0); p.add_argument('--min-positive',type=int,default=20); p.add_argument('--min-negative',type=int,default=50); p.add_argument('--min-weak-positive',type=int,default=10); p.add_argument('--smoke',action='store_true')
    base.run(p.parse_args())

if __name__=='__main__': main()
