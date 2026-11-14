#!/usr/bin/env python3
"""Refresh portable training metrics, keeping raw costs and post-eval costs."""
from pathlib import Path
import csv
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/training/latest_summary.csv'
OUT.parent.mkdir(parents=True, exist_ok=True)
fields='dataset,model,distribution,seed,algorithm,rounds,final_accuracy,best_accuracy,best_round,convergence_round,convergence_round_1based,model_cost,privacy_cost,energy_cost,total_cost,final_test_loss,max_test_loss,status,h5,fair_posteval_model_cost,fair_posteval_total_cost,fair_minus_stored_total_cost,cost_timing_status,checkpoint,training_reuse_status,source_h5,source_checkpoint'.split(',')
rows=[]
if OUT.is_file():
    with OUT.open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
by_path={r['h5']:r for r in rows}
for p in sorted((ROOT/'results').glob('*.h5')):
    rel=p.relative_to(ROOT).as_posix()
    if rel not in by_path:
        row=dict.fromkeys(fields,'');row['h5']=rel;rows.append(row);by_path[rel]=row
errors=[]
for r in rows:
    p=ROOT/r['h5']
    if not p.is_file():continue
    try:
        with h5py.File(p) as f:
            attrs=dict(f.attrs)
            v={k:np.asarray(f[k],dtype=float) for k in ['rs_test_acc','rs_test_loss','round_model_cost','round_privacy_cost','round_energy_cost','round_total_cost']}
        n=len(v['rs_test_acc'])
        if n!=int(attrs['rounds_requested']) or any(a.shape!=(n,) or not np.isfinite(a).all() for a in v.values()):raise ValueError('missing/incomplete/nonfinite arrays')
        if not np.allclose(v['round_total_cost'],v['round_model_cost']+v['round_privacy_cost']+v['round_energy_cost']):raise ValueError('component sum mismatch')
        with h5py.File(p) as f:
            selected=np.asarray(f['round_selected_count']) if 'round_selected_count' in f else np.full(n,int(int(attrs['num_clients'])*float(attrs['join_ratio'])))
        target=float(attrs.get('target_accuracy',.6))
        for k in ['algorithm','dataset','model','seed']:
            r[k]=attrs[k]
        r['distribution']=str(attrs['distribution']);r['rounds']=n
        r['final_accuracy']=float(v['rs_test_acc'][-1]);r['best_accuracy']=float(v['rs_test_acc'].max());r['best_round']=int(v['rs_test_acc'].argmax())+1
        reached=np.flatnonzero(v['rs_test_acc']>=target)
        r['convergence_round']=int(reached[0]) if len(reached) else -1
        r['convergence_round_1based']=int(reached[0])+1 if len(reached) else -1
        for k in ['model','privacy','energy','total']:r[k+'_cost']=float(v['round_'+k+'_cost'].sum())
        fair_model=selected*np.maximum(0,target-v['rs_test_acc'])
        r['fair_posteval_model_cost']=float(fair_model.sum())
        r['fair_posteval_total_cost']=float((fair_model+v['round_privacy_cost']+v['round_energy_cost']).sum())
        r['fair_minus_stored_total_cost']=r['fair_posteval_total_cost']-r['total_cost']
        r['cost_timing_status']='UNIFIED_POST_AGGREGATION_EVAL_OFFLINE'
        r['final_test_loss']=float(v['rs_test_loss'][-1]);r['max_test_loss']=float(v['rs_test_loss'].max())
        bad='fedceo_utility_fail' in r['h5']
        prefix=f"{r['dataset']}_{r['model']}_{r['algorithm']}_"
        label=p.stem[len(prefix):-2]
        cp=ROOT/'system/models'/r['dataset']/r['model']/f"{r['algorithm']}_{label}_0_server.pt"
        if bad:cp=ROOT/'evidence/fedceo_utility_fail/models'/cp.name
        r['checkpoint']=str(cp.relative_to(ROOT)) if cp.is_file() else ''
        r['status']='VALID_H5_UTILITY_FAIL' if bad else 'COMPLETE_H5_CHECK_PASS'
        r['training_reuse_status']='DIAGNOSTIC_UTILITY_FAIL' if bad else ('REUSE_CURVE_AND_MODEL' if r['checkpoint'] else 'REUSE_CURVE_ONLY_CHECKPOINT_MISSING')
    except Exception as e:errors.append(f"{p.name}: {e}")
with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(f'Updated {OUT}; {sum(str(r["rounds"])=="800" for r in rows)} complete numerical records, {len(errors)} rejected.')
for e in errors:print('REJECTED:',e)
if errors:raise SystemExit(1)
