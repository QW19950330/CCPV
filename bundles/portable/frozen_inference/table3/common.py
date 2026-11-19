"""Explicit hard-pinned data/checkpoint inputs and create-only outputs."""
from pathlib import Path
import csv,hashlib,json,os,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def ah(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()
def pin(p):p=Path(p);return dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
def checked(p,r):
    p=Path(p)
    if not p.is_file() or p.stat().st_size!=r["bytes"] or sha(p)!=r["sha256"]:raise ValueError("Input pin mismatch: "+p.name)
    return pin(p)
def put(p,obj):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def save_npz(p,**arrays):
    with Path(p).open("xb") as f:np.savez_compressed(f,**arrays)
def save_csv(p,frame):
    with Path(p).open("x",encoding="utf-8",newline="") as f:frame.to_csv(f,index=False)
def output_dir(p,roots=()):
    p=Path(p).resolve()
    for r in roots:
        if p.is_relative_to(Path(r).resolve()):raise ValueError("Output must be outside input root")
    p.mkdir(parents=False,exist_ok=False);return p
def contract():return read(HERE/"input_contract.json")
def verify_package():
    m=read(HERE/"package_manifest.json")
    for r in m["files"]:checked(HERE/r["file"],r)
    return m["files"]
def check_inputs(args):
    c=contract()
    receipts={k:checked(getattr(args,k),c["data"][k]) for k in ("development_data","development_truth","external_data")}
    receipts["checkpoints"]=[]
    root=args.weights_root.resolve()
    for r in c["checkpoints"]:
        p=(root/r["file"]).resolve()
        if not p.is_relative_to(root):raise ValueError("Checkpoint escaped explicit root")
        receipts["checkpoints"].append(checked(p,r))
    return receipts
def load_development(data_path,truth_path):
    c=contract();checked(data_path,c["data"]["development_data"]);checked(truth_path,c["data"]["development_truth"])
    with np.load(data_path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(truth_path,allow_pickle=False) as z:
        if not np.array_equal(z["source_indices"],a["source_indices"]) or not np.array_equal(z["position_valid"],a["position_valid"]):raise ValueError("Position truth row/mask mismatch")
        a["position_targets"]=z["positions"]
    return a
def validation_rows(a):
    ix=a["validation_indices"]
    return {k:(v.copy() if k=="teams" and v.ndim==1 else v[ix].copy()) for k,v in a.items() if k not in ("train_indices","validation_indices")}
def load_external(path):
    c=contract();checked(path,c["data"]["external_data"])
    with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    return a
def load_model(arm,seed,weights_root,device):
    import torch
    from runtime import make_model
    c=contract();r=next(r for r in c["checkpoints"] if r["arm"]==arm and r["seed"]==seed)
    root=Path(weights_root).resolve();p=(root/r["file"]).resolve()
    if not p.is_relative_to(root):raise ValueError("Checkpoint escaped explicit root")
    checked(p,r) # Verify trusted original bytes before deserializing full original payload.
    payload=torch.load(p,map_location=device,weights_only=False)
    if payload["epoch"]!=4 or payload["arm"]!=arm or payload["seed"]!=seed:raise ValueError("Fixed epoch4 arm/seed differs")
    if payload["protocol_sha256"]!=c["original_protocol_sha256"]:raise ValueError("Checkpoint protocol differs")
    model=make_model(arm,seed);model.load_state_dict(payload["state_dict"],strict=True)
    return model.to(device).eval()
