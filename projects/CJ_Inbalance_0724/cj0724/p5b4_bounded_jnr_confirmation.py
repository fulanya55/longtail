"""Read-only two-new-identity confirmation for the frozen P5b3 attachment."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from cj0724.p5b1_jnr_physical_oracle import FeatureProbe, PairedPowerOracleDataset, _power_files, bootstrap_gain, metrics, sha256_file
from cj0724.p5b2_posterior_conditioned_jnr import collect_analytic_role, collect_frozen_posterior, with_context
from cj0724.p5b3_bounded_jnr_attachment import TARGET_CLASSES, TARGET_NAMES, collect_base_parameter, load_base, predicted_gated_role

def load_head(path: Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = FeatureProbe(int(state["input_dim"])).to(device); model.load_state_dict(state["model_state"]); model.eval()
    return model, state

def estimate(role, model, state, device):
    x = np.concatenate([role.feature, role.composition], axis=1).astype(np.float32)
    x = (x - state["feature_mean"]) / state["feature_scale"]
    with torch.inference_mode():
        value = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return value * state["target_scale"] + state["target_mean"]

def evaluate_root(project, root, protocol, audit):
    files = _power_files(root); reference = PairedPowerOracleDataset(files, np.arange(3780, dtype=np.int64))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analytic = collect_analytic_role(reference, device); rows=[]
    for source_seed, head_path in zip(protocol["frozen_candidate"]["base_model_seeds"], protocol["frozen_candidate"]["head_checkpoints"]):
        model, normalizer, thresholds, source = load_base(project, int(source_seed)); model=model.to(device).eval()
        posterior=collect_frozen_posterior(reference, model, device); role=with_context(analytic, posterior)
        head, state=load_head(project / head_path, device); value=estimate(role, head, state, device)
        logits, base_full=collect_base_parameter(reference, model, normalizer, device); probability=1/(1+np.exp(-logits)); decision=probability>=thresholds[None,:]
        base=base_full[:,TARGET_CLASSES,0].copy(); attached_full=base_full.copy()
        for local, cls in enumerate(TARGET_CLASSES): attached_full[decision[:,cls],cls,0]=value[decision[:,cls],local]
        attached=attached_full[:,TARGET_CLASSES,0]; allowed=np.zeros_like(base_full,bool)
        for cls in TARGET_CLASSES: allowed[:,cls,0]=decision[:,cls]
        rows.append({"source":source,"head":str(project/head_path),"head_sha256":sha256_file(project/head_path),
            "base":metrics(base,role,np.asarray(state["target_scale"])),"attached":metrics(attached,role,np.asarray(state["target_scale"])),
            "predicted_base":metrics(base,predicted_gated_role(role,probability,thresholds),np.asarray(state["target_scale"])),
            "predicted_attached":metrics(attached,predicted_gated_role(role,probability,thresholds),np.asarray(state["target_scale"])),
            "invariant":bool(np.array_equal(base_full[~allowed],attached_full[~allowed])),"target_scale":np.asarray(state["target_scale"]),"base_prediction":base,"attached_prediction":attached,"role":role})
        del model, head
        if device.type=="cuda": torch.cuda.empty_cache()
    reference.close(); return rows

def summarize(rows):
    summary={}; bootstrap={}; failures=[]
    for local,name in enumerate(TARGET_NAMES):
        gain=[]; pred_delta=[]
        for row in rows:
            old=row["base"]["by_target"][name]["nmae"]; new=row["attached"]["by_target"][name]["nmae"]; gain.append(1-new/old)
            pred_delta.append(row["predicted_attached"]["by_target"][name]["nmae"]-row["predicted_base"]["by_target"][name]["nmae"])
            for power in (10,30):
                if row["attached"]["by_target"][name]["per_power_nmae"][str(power)]-row["base"]["by_target"][name]["per_power_nmae"][str(power)]>0.03: failures.append({"target":name,"power":power})
        # All five views remain together inside each identity cluster.
        bootstrap[name]=[bootstrap_gain(r["base_prediction"],r["attached_prediction"],r["role"],local,r["target_scale"],14041+i) for i,r in enumerate(rows)]
        summary[name]={"gain_by_model":gain,"predicted_gated_delta_by_model":pred_delta}
    return summary,bootstrap,failures

def main():
    a=argparse.ArgumentParser();a.add_argument('--preregister',required=True);a.add_argument('--data-roots',nargs='+',required=True);a.add_argument('--audits',nargs='+',required=True);a.add_argument('--output-root',required=True);x=a.parse_args();project=Path.cwd(); protocol=json.loads((project/x.preregister).read_text(encoding='utf-8'))
    if len(x.data_roots)!=2 or len(x.audits)!=2: raise RuntimeError('P5b4 requires exactly two fresh audits')
    output={"experiment":protocol["experiment"],"preregister_sha256":sha256_file(project/x.preregister),"data":{},"test_generated":False,"test_opened":False,"locked_test_used":False}
    all_pass=True
    for root,audit_path in zip(x.data_roots,x.audits):
        audit=json.loads((project/audit_path).read_text(encoding='utf-8'))
        if not audit["valid_for_frozen_inference"]: raise RuntimeError('invalid confirmation audit')
        rows=evaluate_root(project,project/root,protocol,audit); summary,boot,failures=summarize(rows)
        gates={"invariance":all(r["invariant"] for r in rows),"mft_positive":all(v>0 for v in summary["MFT.JNR"]["gain_by_model"]),"isrj_positive":all(v>0 for v in summary["ISRJ.JNR"]["gain_by_model"]),"mft_predicted":all(v<=0 for v in summary["MFT.JNR"]["predicted_gated_delta_by_model"]),"isrj_predicted":all(v<=0 for v in summary["ISRJ.JNR"]["predicted_gated_delta_by_model"]),"bootstrap":all(all(v["ci95_low"]>0 for v in boot[n]) for n in ("MFT.JNR","ISRJ.JNR")),"power":not failures}
        all_pass &= all(gates.values()); output["data"][root]={"audit":audit,"rows":[{k:v for k,v in r.items() if k not in ("base_prediction","attached_prediction","role","target_scale")} for r in rows],"summary":summary,"bootstrap":boot,"power_failures":failures,"gates":gates}
    output["confirmation_passed"]=bool(all_pass);out=project/x.output_root;out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps({"confirmation_passed":all_pass,"gates":{k:v['gates'] for k,v in output['data'].items()}},indent=2))
if __name__=='__main__':main()
