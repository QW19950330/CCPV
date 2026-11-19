"""Frozen same-format test-cohort response; no training or future-label access."""
from pathlib import Path
import argparse, collections, hashlib, importlib, json, os, sys, time, traceback, types
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ORIGINAL = BASE / "CCPV_reviewer_validation_program_20261026/full_intervention"
PROTOCOL = HERE / "skillcorner_response_protocol.json"
PATHS = ("natural", "none_every_step", "first_A", "persistent_A",
         "maintained_v0_A", "first_provenance_only_A")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(clean(x), f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")


def npz(p, **a):
    with Path(p).open("xb") as f: np.savez_compressed(f, **a)


def runtime():
    import torch
    sys.path.insert(0, str(ORIGINAL))
    sys.path.insert(0, str(ORIGINAL / "model_source"))
    api = importlib.import_module("api")
    adapter = importlib.import_module("operator_adapter")
    assert Path(api.__file__).resolve() == (ORIGINAL / "model_source/api.py").resolve()
    assert Path(adapter.__file__).resolve() == (ORIGINAL / "operator_adapter.py").resolve()
    def forbidden(*a, **kw):
        raise RuntimeError("Mixed source data or implicit model loader forbidden")
    api.load_development = api.load_validation = api.load_model = forbidden
    api.configure("cpu")
    return torch, api, adapter


def bits(a,b):
    a = np.ascontiguousarray(a); b = np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def sham_api(api, slots):
    def initial(valid):
        observed = valid.clone()
        for row,slot in enumerate(slots):
            if slot >= 0: observed[row,-1,int(slot)] = False
        _, age = api.initial_provenance(observed)
        return observed, age
    return types.SimpleNamespace(DT=api.DT, initial_provenance=initial,
        current_anchor=api.current_anchor, step=api.step, advance_provenance=api.advance_provenance)


def paths(torch, api, adapter, model, h, v, tm, registry, targets, slots, protocol):
    limits = adapter.MotionLimits(protocol["limits"]["vmax"],
                                  protocol["limits"]["amax"], protocol["limits"]["turnmax"])
    common = dict(units=adapter.MetricUnits("m","m/s","s","m/s^2","rad/s"), dt=.1, horizon=20)
    command = adapter.make_commands(h, registry, targets)
    v0 = adapter.make_commands(h, registry, targets, delta0=True)
    results = {}
    specifications = [
        ("natural",api,None,(),"bypass"),
        ("none_every_step",api,None,(),"none"),
        ("first_A",api,command,(0,),"first"),
        ("persistent_A",api,command,(0,),"persistent"),
        ("maintained_v0_A",api,v0,(0,),"persistent"),
        ("first_provenance_only_A",sham_api(api,slots),None,(),"bypass"),
    ]
    for name, runtime_api, cmd, coalition, semantic in specifications:
        results[name] = adapter.rollout(runtime_api, model, h, v, tm, registry, targets,
                                       cmd, coalition, semantic, limits, **common)
    if not bits(results["natural"]["states"].numpy(), results["none_every_step"]["states"].numpy()):
        raise AssertionError("NONE not bitwise identical to natural")
    if not bits(results["natural"]["support"].numpy(), results["none_every_step"]["support"].numpy()):
        raise AssertionError("NONE support differs")
    # Original factual mean interface versus adapter, only complete natural rows.
    active = results["natural"]["initial_scene_supported"]
    if active.any():
        if model is None:
            expected = api.analytic_forecast("cv", h[active], v[active])
            tolerance = 1e-4
        else:
            with torch.inference_mode():
                expected = api.predict(model, h[active], v[active], tm if tm.ndim==1 else tm[active])
            tolerance = 1e-6
        observed = results["natural"]["states"][active]
        if not torch.isfinite(expected).all() or (expected-observed).abs().max().item() > tolerance:
            raise AssertionError("Native and adapter factual paths differ")
    if model is None:
        if not bits(results["first_provenance_only_A"]["states"].numpy(),results["natural"]["states"].numpy()):
            raise AssertionError("CV provenance-only sham differs from natural")
        for row,slot in enumerate(slots):
            if slot < 0 or results["first_A"]["completed_steps"][row] != 20: continue
            elapsed = torch.arange(1,21,dtype=h.dtype)[:,None] * .1
            delta = command[row,0] - v0[row,0]
            actual = results["first_A"]["states"][row,:,slot,:2] - results["natural"]["states"][row,:,slot,:2]
            if (actual-elapsed*delta).abs().max().item() > 1e-4:
                raise AssertionError("CV analytic t*delta differs")
            if results["persistent_A"]["completed_steps"][row] == 20:
                if (results["first_A"]["states"][row] - results["persistent_A"]["states"][row]).abs().max().item() > 1e-4:
                    raise AssertionError("CV first and persistent disagree")
    return results, command.numpy(), v0.numpy()


def compact(results, take, size, hvalid, slots, command, v0):
    states = np.full((size,6,20,23,4),np.nan,np.float32)
    support = np.zeros((size,6,20,23),bool)
    steps = np.zeros((size,6),np.int64)
    status = np.full((size,6),"UNSUPPORTED_PREPARED",dtype="<U64")
    reason = [["Prepared input or actor unsupported"]*6 for _ in range(size)]
    checks = {}
    for k,name in enumerate(PATHS):
        if not len(take): continue
        r = results[name]
        states[take,k] = r["states"].numpy(); support[take,k] = r["support"].numpy()
        steps[take,k] = r["completed_steps"].numpy(); status[take,k] = r["scene_status"]
        for j,original in enumerate(take): reason[int(original)][k] = r["scene_reasons"][j]
        checks[name] = dict(checks=r["checks"], effective=int(r["scene_effective_forward_counts"].sum()),
            dummy=int(r["scene_dummy_forward_counts"].sum()), operator_calls=r["actual_operator_calls"],
            rejected=len(r["rejections"]), observed_flags_cleared=len(r["edited_observation_flags_cleared"]))
    direction = np.full((size,2),np.nan,np.float32)
    direction[take] = command[:,0] - v0[:,0]
    direction[take] /= np.linalg.norm(direction[take],axis=1)[:,None]
    # Four full vector differences plus the two first-step provenance controls.
    pairs = ((2,0),(3,0),(4,0),(5,0),(2,5),(3,4))
    actor_vector = np.full((size,6,20,2),np.nan,np.float32)
    joint_rms = np.full((size,6,20),np.nan,np.float32)
    for b in range(size):
        slot = int(slots[b])
        if slot < 0: continue
        historical = hvalid[b].any(0).copy(); historical[slot]=False; historical[22]=False
        for k,(left,right) in enumerate(pairs):
            ok = support[b,left,:,slot] & support[b,right,:,slot]
            actor_vector[b,k,ok] = states[b,left,ok,slot,:2] - states[b,right,ok,slot,:2]
            for t in range(20):
                keep=historical & support[b,left,t] & support[b,right,t]
                if keep.any():
                    d=states[b,left,t,keep,:2]-states[b,right,t,keep,:2]
                    joint_rms[b,k,t]=np.sqrt(np.mean(np.sum(d*d,axis=-1)))
    projected = np.sum(actor_vector*direction[:,None,None,:],axis=-1)
    return dict(states=states,support=support,completed_steps=steps,status=status,
                actor_difference_vectors=actor_vector,actor_distance=np.linalg.norm(actor_vector,axis=-1),
                actor_direction_projection=projected,other_player_rms=joint_rms), reason, checks


def validate_resume_block(old, bp, *, arm, seed, start, stop, input_sha, implementation_sha, expected_rows):
    """Bind a complete saved block to its actual cell, interval and prepared rows."""
    expected = dict(arm=arm, seed=seed, start=start, stop=stop, path=Path(bp).name,
                    input_sha256=input_sha, implementation_sha256=implementation_sha)
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError("Resume block identity mismatch: " + key)
    if old.get("sha256") != sha(bp):
        raise ValueError("Resume block payload hash mismatch")
    with np.load(bp, allow_pickle=False) as saved:
        for key, value in expected_rows.items():
            if key not in saved.files or not bits(saved[key], value):
                raise ValueError("Resume block prepared-row mismatch: " + key)


def validate_cell_completion(old, current):
    for key in ("cell", "planned_rows", "batches", "parameter_hash_before",
                "parameter_hash_after", "checkpoint", "blocks"):
        if old.get(key) != current[key]:
            raise ValueError("Saved cell completion mismatch: " + key)


def verify_lock():
    lock = read(HERE/"implementation_lock.json")
    for r in lock["pins"]:
        p = Path(r["path"]); p = p if p.is_absolute() else BASE/p
        if sha(p) != r["sha256"]: raise ValueError("Frozen input/code changed: "+str(p))
    return lock


def synthetic(receipt=None):
    torch,api,adapter=runtime(); protocol=read(PROTOCOL)
    h=torch.zeros((3,10,23,4),dtype=torch.float32)
    h[...,0]=torch.arange(23)[None,None,:]; h[...,2]=1.
    h[:,:,:,0] += torch.arange(10)[None,:,None]*.1
    v=torch.ones((3,10,23),dtype=torch.bool); v[1,:,22]=False; h[1,:,22]=0
    v[2,-2:,3]=False; h[2,-2:,3]=0
    tm=torch.tensor([0]*11+[1]*11+[2],dtype=torch.long)
    ids=[[f"synthetic-player-{i}" for i in range(22)]+["synthetic-ball"] for _ in range(3)]
    types_=[["VERIFIED_SOURCE_PLAYER_TEAM"]*22+["EXPLICIT_MODEL_BALL_SOURCE_ID"] for _ in range(3)]
    registry=adapter.make_registries(ids,types_,tm)
    targets=[[ids[b][0],ids[b][1]] for b in range(3)]
    rr,command,v0=paths(torch,api,adapter,None,h,v,tm,registry,targets,[0,0,0],protocol)
    assert torch.equal(rr["first_provenance_only_A"]["states"],rr["natural"]["states"])
    assert not rr["first_provenance_only_A"]["final_observation_valid"].any()
    model=api.make_model("D_coupled_ar",0).eval()  # Random synthetic weights; no saved checkpoint/data read.
    nr,_,_=paths(torch,api,adapter,model,h,v,tm,registry,targets,[0,0,0],protocol)
    assert all(all(r["checks"].values()) for r in nr.values())
    sa=sham_api(api,[0,0,0]); observed,age=sa.initial_provenance(v)
    assert not observed[:,-1,0].any() and torch.equal(observed[:,:-1],v[:,:-1])
    assert torch.equal(observed[:,-1,1:],v[:,-1,1:])
    full, reasons,checks=compact(rr,np.arange(3),4,
        np.concatenate((v.numpy(),np.zeros((1,10,23),bool))),[0,0,0,-1],command,v0)
    assert np.isnan(full["actor_distance"][3]).all() and not full["support"][3].any()
    write(Path(receipt) if receipt else HERE/"runner_synthetic.json",dict(status="PASS_CV_SHAM_MISSING_SUPPORT_AND_RANDOM_D",
          code_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),synthetic_rows=3,
          saved_checkpoint_loads=0,real_prefix_reads=0,training_updates=0))


def run():
    lock=verify_lock(); protocol=read(PROTOCOL)
    torch,api,adapter=runtime()
    a_path=HERE/"skillcorner_prepared/test_prefixes.npz"
    input_sha = sha(a_path)
    if input_sha != lock["prepared_prefix_sha256"]:
        raise ValueError("Prepared prefix SHA differs from implementation lock")
    implementation_sha = sha(HERE/"implementation_lock.json")
    with np.load(a_path,allow_pickle=False) as z: a={k:z[k] for k in z.files}
    if a["history_states"].shape != (4992,10,23,4): raise ValueError("Full fixed4992 required")
    assets=read(ORIGINAL/"assets.json")
    cells=[dict(arm="CV",seed=-1,checkpoint=None)]
    cells += [dict(arm=r["arm"],seed=r["seed"],checkpoint=r) for r in assets["checkpoints"] if r["arm"]=="D_coupled_ar"]
    assert [c["seed"] for c in cells[1:]] == list(range(5))
    out=HERE/"response_run01"; out.mkdir(exist_ok=True)
    receipts=[]
    for cell in cells:
        cell_started=time.monotonic(); name=f'{cell["arm"]}_s{cell["seed"]}'; folder=out/name
        folder.mkdir(exist_ok=True)
        model=None
        if cell["checkpoint"]:
            cp=cell["checkpoint"]; assert sha(cp["path"])==cp["sha256"]
            payload=torch.load(cp["path"],map_location="cpu",weights_only=False)
            assert payload["epoch"]==4
            model=api.make_model(cell["arm"],cell["seed"]); model.load_state_dict(payload["state_dict"],strict=True); model.eval()
        before=None if model is None else api.state_hash(model.state_dict())
        batch_receipts=[]
        for start in range(0,4992,32):
            stop=min(start+32,4992); bp=folder/f"batch_{start:05d}.npz"; rp=folder/f"batch_{start:05d}.json"
            if rp.exists():
                old=read(rp)
                validate_resume_block(old, bp, arm=cell["arm"], seed=cell["seed"],
                    start=start, stop=stop, input_sha=input_sha, implementation_sha=implementation_sha,
                    expected_rows=dict(source_indices=a["source_indices"][start:stop],
                        match_ids=a["match_ids"][start:stop],
                        actor_slots=a["operator_candidate_slots"][start:stop,0],
                        technical_unsupported=a["technical_unsupported"][start:stop]))
                batch_receipts.append(old); continue
            if bp.exists(): raise FileExistsError("Incomplete orphan block must be reviewed, not overwritten")
            hs=a["history_states"][start:stop]; vv=a["history_valid"][start:stop]
            slots=a["operator_candidate_slots"][start:stop,0]
            take=np.flatnonzero((slots>=0) & ~a["technical_unsupported"][start:stop])
            batch_checks={}; reasons=[["Prepared input or actor unsupported"]*6 for _ in range(stop-start)]
            if len(take):
                regs=adapter.make_registries(a["entity_ids"][start:stop][take],a["identity_status"][start:stop][take],a["teams"])
                targets=a["operator_candidate_ids"][start:stop][take].tolist()
                h=torch.from_numpy(hs[take].copy()); v=torch.from_numpy(vv[take].copy()); tm=torch.from_numpy(a["teams"].copy())
                r,cmd,v0=paths(torch,api,adapter,model,h,v,tm,regs,targets,slots[take].tolist(),protocol)
                arrays,reasons,batch_checks=compact(r,take,stop-start,vv,slots,cmd,v0)
            else:
                arrays,reasons,batch_checks=compact({},take,stop-start,vv,slots,np.empty((0,2,2),np.float32),np.empty((0,2,2),np.float32))
            arrays.update(source_indices=a["source_indices"][start:stop],
                match_ids=a["match_ids"][start:stop],actor_slots=slots,
                technical_unsupported=a["technical_unsupported"][start:stop])
            npz(bp,**arrays)
            rec=dict(start=start,stop=stop,path=bp.name,sha256=sha(bp),
                implementation_sha256=implementation_sha,
                input_sha256=input_sha,
                eligible_initial_rows=len(take),status_counts=dict(collections.Counter(arrays["status"].ravel())),
                checks=batch_checks,reasons=reasons,arm=cell["arm"],seed=cell["seed"])
            write(rp,rec); batch_receipts.append(rec)
            print(json.dumps(dict(cell=name,completed_rows=stop,total=4992,seconds=time.monotonic()-cell_started)),flush=True)
            if time.monotonic()-cell_started>protocol["runtime"]["soft_seconds_per_cell"]:
                raise TimeoutError("Cell runtime boundary reached; saved whole batches retained")
        after=None if model is None else api.state_hash(model.state_dict())
        if before!=after: raise AssertionError("Frozen model changed")
        cr=dict(cell=name,planned_rows=4992,batches=len(batch_receipts),parameter_hash_before=before,
                parameter_hash_after=after,checkpoint=cell["checkpoint"],seconds=time.monotonic()-cell_started,
                blocks=[dict(path=r["path"],sha256=r["sha256"]) for r in batch_receipts])
        if (folder/"completion.json").exists():
            validate_cell_completion(read(folder/"completion.json"), cr)
        else: write(folder/"completion.json",cr)
        receipts.append(cr)
    verify_lock()
    final = dict(status="COMPLETE_FIXED4992_SIX_CELLS_SIX_PATHS",
          cells=receipts,paths=PATHS,planned_source_windows=4992,new_training_updates=0,
          future_truth_read=False,original_sources_unchanged=True)
    if (out/"completion.json").exists():
        old_final=read(out/"completion.json")
        for key in ("status","paths","planned_source_windows","new_training_updates",
                    "future_truth_read","original_sources_unchanged"):
            if old_final.get(key) != clean(final[key]):
                raise ValueError("Saved run completion mismatch: " + key)
        if len(old_final.get("cells",[])) != len(receipts):
            raise ValueError("Saved run completion cell count mismatch")
        for old_cell,new_cell in zip(old_final["cells"],receipts):
            validate_cell_completion(old_cell,new_cell)
    else: write(out/"completion.json",final)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["synthetic","run"])
    p.add_argument("--synthetic-receipt"); args=p.parse_args()
    if args.stage=="run" and args.synthetic_receipt:
        p.error("--synthetic-receipt applies only to synthetic stage")
    try:
        if args.stage=="synthetic": synthetic(args.synthetic_receipt)
        else: run()
    except Exception as e:
        if args.stage=="run":
            dst=HERE/"response_run01"/("failure_"+time.strftime("%Y%m%dT%H%M%S")+".json")
            write(dst,dict(error=repr(e),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
