"""Actual random forest on summaries of revealed sensor observations."""
import argparse,time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from .method import Model


def summaries(values,mask):
    """Per-channel revealed mean/std/min/max/last, observed fraction and elapsed mean."""
    mask=mask.bool();count=mask.sum(1).clamp_min(1);zero=torch.zeros_like(values[:,0])
    mean=torch.where(mask,values,0.).sum(1)/count
    std=(torch.where(mask,(values-mean[:,None]).square(),0.).sum(1)/count).sqrt()
    minimum=torch.where(mask,values,torch.inf).amin(1);maximum=torch.where(mask,values,-torch.inf).amax(1)
    minimum=torch.where(mask.any(1),minimum,zero);maximum=torch.where(mask.any(1),maximum,zero)
    last=zero.clone();gap=zero.clone();elapsed=zero.clone()
    for step in range(values.shape[1]):
        last=torch.where(mask[:,step],values[:,step],last)
        gap=(gap+1)*(~mask[:,step]);elapsed+=gap/values.shape[1]
    return torch.cat([mean,std,minimum,maximum,last,mask.float().mean(1),elapsed/values.shape[1]],1).cpu().numpy()


class ForestModel(Model):
    def __init__(self,config):
        super().__init__(dict(config,backbone='tcn'))
        self.c=dict(config);self.encoder=nn.Identity();self.head=nn.Identity();self.forest=None
    def classify(self,x,mask):
        features=summaries(self.normalized(x),mask)
        probability=np.zeros((len(features),len(self.class_weights)))
        probability[:,self.forest.classes_.astype(int)]=self.forest.predict_proba(features)
        self.last_auxiliary=x.new_zeros(())
        return torch.as_tensor(probability,device=x.device,dtype=x.dtype).clamp_min(1e-10).log()


def main():
    from .cli import read,tensor_batch,dump,validate
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--trees',type=int,default=200);parser.add_argument('--max-depth',type=int)
    parser.add_argument('--seed',type=int,default=13);parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args();torch.set_num_threads(1)
    root,output=Path(args.input),Path(args.output);output.mkdir(parents=True,exist_ok=True)
    data,validation=read(root/'train.npz'),read(root/'validation.npz')
    if np.intersect1d(data['group'],validation['group']).size:raise ValueError('Fitting/development groups overlap')
    config=dict(input_shape=list(data['x'].shape[1:]),classes=int(data['y'].max())+1,backbone='random_forest',variant='baseline',seed=args.seed)
    model=ForestModel(config);model.prepare(data);features=[];labels=[];torch.manual_seed(args.seed)
    started=time.perf_counter()
    for start in range(0,len(data['x']),args.batch_size):
        ids=np.arange(start,min(start+args.batch_size,len(data['x'])))
        batch=tensor_batch(data,ids,'cpu');raw=batch['x'].float();native=batch.get('mask',torch.isfinite(raw)).bool()&torch.isfinite(raw)
        for condition in range(5):
            mask=native if condition==0 else model.policy(torch.nan_to_num(raw),condition-1,native)
            features.append(summaries(model.normalized(raw),mask));labels.append(data['y'][ids])
    model.forest=RandomForestClassifier(n_estimators=args.trees,max_depth=args.max_depth,n_jobs=1,random_state=args.seed,class_weight='balanced')
    model.forest.fit(np.concatenate(features),np.concatenate(labels));model.eval()
    clean,risks=validate(model,validation,args.batch_size,'cpu',args.seed)
    joblib.dump(model.forest,output/'model.joblib')
    torch.save(dict(config=config,state=model.state_dict(),training_groups=np.unique(data['group']).tolist(),validation_groups=np.unique(validation['group']).tolist()),output/'model.pt')
    dump(output/'training.json',dict(seconds=time.perf_counter()-started,trees=args.trees,max_depth=args.max_depth,seed=args.seed,training_rows=len(np.concatenate(labels)),validation_clean_ce=clean,validation_worst_policy_ce=float(risks.max())))


if __name__=='__main__':main()
