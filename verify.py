"""CPU functional checks; generated observations do not estimate paper performance."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from missingness_games import cli, evaluate_policies
from missingness_games.data import har, pamap2
from missingness_games.method import Model, VARIANTS, demo
from missingness_games.metrics import measure


def invoke(*args):
    with patch.object(sys,'argv',['missingness-games',*map(str,args)]):cli.main()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/functional-check')
    root=Path(parser.parse_args().output);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(13)
    config=dict(input_shape=[16,6],classes=3,width=8,variant='full',clean_guard_tolerance=2.,
        channel_groups=[[0,1,2],[3,4,5]],fixed_hard_index=0,memory=.9,temperature=.2)
    raw=demo(13,60,config);raw['group']=np.arange(60)//6
    raw['split']=np.where(raw['group']<6,0,np.where(raw['group']==6,1,np.where(raw['group']==7,2,3)))
    raw['x'][0,0,0]=np.nan
    np.savez_compressed(root/'input.npz',**raw)
    invoke('prepare','--input',root/'input.npz','--output',root/'data')
    data=cli.read(root/'data/train.npz');batch=cli.tensor_batch(data,np.arange(6),'cpu')
    model=Model(config);model.prepare(data)
    np.testing.assert_allclose(model.mean.numpy(),np.nanmean(data['x'],axis=(0,1)),rtol=1e-5)
    assert not torch.isfinite(batch['x']).all()
    for index in range(6):
        complete=torch.nan_to_num(batch['x']);observed=torch.ones_like(complete,dtype=torch.bool)
        torch.manual_seed(42);mask=model.policy(complete,index,observed)
        deleted=(1-mask).sum((1,2));assert (deleted<=int(16*6*.4)).all()
        gap=torch.zeros(6,6);maximum=0
        for t in range(16):
            gap=(gap+1)*(1-mask[:,t]);maximum=max(maximum,int(gap.max()))
        assert maximum<=4
        # A current value cannot affect the action chosen to hide it.
        altered=complete.clone();altered[:,8:]+=10000
        torch.manual_seed(42);changed=model.policy(altered,index,observed)
        torch.testing.assert_close(mask[:,:9],changed[:,:9])
        # Hidden values cannot influence any later history-dependent action.
        altered=torch.where(mask.bool(),complete,complete+10000)
        torch.manual_seed(42);changed=model.policy(altered,index,observed)
        torch.testing.assert_close(mask,changed)
    fallback=Model(dict(config, independent_probability=1., max_gap=10))
    fallback_mask=fallback.policy(torch.ones(1,4,6),0)
    assert (fallback_mask[:,0]==0).all() and (fallback_mask[:,1:]==1).all()
    model.train();prediction=model(batch)
    expected_losses=torch.stack([model.cross_entropy(prediction['policy_logits'][:,k],batch['y']).mean() for k in range(4)])
    before=model.encoder[0].weight.detach().clone();loss=model.objective(batch,prediction)
    torch.testing.assert_close(model.memory,expected_losses.detach())
    expected_weights=(model.reference.log()+expected_losses.detach()/.2).softmax(0)
    np.testing.assert_allclose(model.last_diagnostics['weights'],expected_weights.numpy(),rtol=1e-6)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01);optimizer.zero_grad();loss.backward()
    assert torch.isfinite(loss) and model.encoder[0].weight.grad.abs().sum()>0 and model.head.weight.grad.abs().sum()>0
    optimizer.step();assert not torch.equal(before,model.encoder[0].weight)
    first=model.memory.clone();prediction=model(batch)
    second=torch.stack([model.cross_entropy(prediction['policy_logits'][:,k],batch['y']).mean() for k in range(4)])
    model.objective(batch,prediction);torch.testing.assert_close(model.memory,.9*first+.1*second.detach())
    model.eval();expected=model(batch)['probability'].detach();memory=model.memory.clone()
    torch.save(model.state_dict(),root/'state.pt');restored=Model(config).eval();restored.load_state_dict(torch.load(root/'state.pt',weights_only=True))
    torch.testing.assert_close(expected,restored(batch)['probability'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(memory,model.memory)
    (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
    invoke('train','--input',root/'data','--output',root/'baseline','--config',root/'config.json','--variant','baseline','--epochs',1)
    invoke('pilot','--input',root/'data/train.npz','--model',root/'baseline/model.pt','--output',root/'pilot.json')
    for variant in VARIANTS:
        if variant=='baseline':continue
        invoke('train','--input',root/'data','--output',root/variant,'--config',root/'config.json',
            '--variant',variant,'--epochs',1,'--batch-size',12,'--reference',root/'baseline/model.pt','--pilot',root/'pilot.json')
        history=json.loads((root/variant/'history.json').read_text())
        assert np.isfinite(history[0]['training_loss']) and history[0]['eligible']
    guarded=torch.load(root/'full/model.pt',weights_only=True)['state']
    unguarded=torch.load(root/'unguarded/model.pt',weights_only=True)['state']
    for key in guarded:torch.testing.assert_close(guarded[key],unguarded[key],rtol=0,atol=0)
    full_history=json.loads((root/'full/history.json').read_text())[0]['allocation']
    permuted_history=json.loads((root/'permuted/history.json').read_text())[0]['allocation']
    assert [row['introduced_missing_rates'] for row in full_history]==[row['introduced_missing_rates'] for row in permuted_history]
    invoke('evaluate','--input',root/'data/test.npz','--model',root/'full/model.pt','--output',root/'evaluation')
    invoke('calibrate','--input',root/'data/calibration.npz','--model',root/'full/model.pt','--output',root/'calibrated.pt')
    invoke('evaluate','--input',root/'data/test.npz','--model',root/'calibrated.pt','--output',root/'calibrated_evaluation')
    with patch.object(sys,'argv',['evaluate_policies','--input',str(root/'data/test.npz'),'--model',str(root/'full/model.pt'),'--output',str(root/'policies')]):
        evaluate_policies.main()
    summary=json.loads((root/'policies/summary.json').read_text());assert len(summary['conditions'])==7
    with patch.object(sys,'argv',['evaluate_policies','--input',str(root/'data/test.npz'),'--model',str(root/'full/model.pt'),'--output',str(root/'shuffled'),'--shuffle-masks']):
        evaluate_policies.main()
    a=np.load(root/'policies/channel_burst.npz')['mask'];b=np.load(root/'shuffled/channel_burst.npz')['mask']
    np.testing.assert_equal(a.sum(1),b.sum(1))
    panel=json.loads((root/'policies/panels.json').read_text())
    assert len(panel['panels'])==6 and panel['training_native_boundary']==1.
    assert panel['rare_classes']==[0]
    from missingness_games.panels import panels
    example=dict(x=np.ones((4,2,1)),y=np.array([0,1,0,1]),group=np.array([1,1,2,2]))
    perfect=np.eye(2)[example['y']];bad=perfect[::-1]
    handmade=panels(dict(clean=perfect,bank=perfect,heldout=bad),example,1.,[0],['bank'],['heldout'])
    high=next(row for row in handmade['panels'] if row['completeness']=='higher' and row['rarity']=='all')
    assert high['clean']==1 and high['heldout_worst']==0
    metric=measure(dict(probability=np.array([[.8,.2],[.4,.6]])),dict(y=np.array([0,1]),group=np.array([0,1])))
    assert abs(metric['brier']-.2)<1e-12 and abs(metric['ece']-.3)<1e-12 and metric['subject_macro_accuracy']==1
    # Released file-layout readers are checked with small schema fixtures.
    har_root=root/'har'
    for role,subjects in [('train',[1,7]),('test',[2,3])]:
        folder=har_root/role/'Inertial Signals';folder.mkdir(parents=True,exist_ok=True)
        for name in ('total_acc','body_gyro'):
            for axis in ('x','y','z'):np.savetxt(folder/f'{name}_{axis}_{role}.txt',np.ones((2,8)))
        np.savetxt(har_root/role/f'y_{role}.txt',[1,2],fmt='%d');np.savetxt(har_root/role/f'subject_{role}.txt',subjects,fmt='%d')
    exported,groups=har(har_root);assert exported['x'].shape==(4,8,6) and set(exported['split'])=={0,1,2}
    protocol=root/'pamap/Protocol';protocol.mkdir(parents=True,exist_ok=True)
    for subject in range(101,109):
        table=np.zeros((16,54));table[:,0]=np.arange(16)*.01;table[:,1]=1;table[0,4]=np.nan
        np.savetxt(protocol/f'subject{subject}.dat',table)
    exported,groups=pamap2(root/'pamap',window=8,stride=4)
    assert exported['x'].shape[-1]==18 and not exported['mask'].all()
    assert set(exported['split'])=={0,1,2}
    invoke('plot','--input',root/'evaluation','--output',root/'figures')
    print(json.dumps(dict(status='passed',variants=len(VARIANTS),epochs=1,policy_causality=True,
        checkpoint_equal=True,metric_example=True,raw_schema_readers=True,functional_guard_tolerance=2.)))


if __name__=='__main__':main()
