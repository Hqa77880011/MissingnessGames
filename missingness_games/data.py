"""Export UCI HAR and PAMAP2 released signals without fitting transforms."""
import argparse
import json
from pathlib import Path

import numpy as np

PAMAP_ACTIVITIES = [1,2,3,4,5,6,7,12,13,16,17,24]


def cap_by_subject_activity(x, y, subject, cap, rng):
    selected = []
    for group in np.unique(subject):
        for label in np.unique(y[subject == group]):
            ids = np.flatnonzero((subject == group) & (y == label))
            selected.extend(rng.choice(ids, min(len(ids), cap), replace=False))
    ids = np.sort(selected)
    return x[ids], y[ids], subject[ids]


def har(root, seed=13, cap=50):
    """Use total acceleration/gyro only; reserve training subjects 7/14/22."""
    arrays = []
    rng = np.random.default_rng(seed)
    for role in ('train','test'):
        signals = [np.loadtxt(root/role/'Inertial Signals'/f'{name}_{axis}_{role}.txt', dtype=np.float32, ndmin=2)
                   for name in ('total_acc','body_gyro') for axis in ('x','y','z')]
        x = np.stack(signals, -1)
        y = np.loadtxt(root/role/f'y_{role}.txt', dtype=np.int64, ndmin=1) - 1
        group = np.loadtxt(root/role/f'subject_{role}.txt', dtype=np.int64, ndmin=1)
        x,y,group = cap_by_subject_activity(x,y,group,cap,rng)
        split = np.full(len(y), 2 if role=='test' else 0)
        if role=='train':split[np.isin(group,[7,14,22])] = 1
        arrays.append(dict(x=x,y=y,group=group,split=split))
    return {key:np.concatenate([part[key] for part in arrays]) for key in arrays[0]}, [[0,1,2],[3,4,5]]


def pamap2(root, test_subject=101, seed=13, cap=100, window=256, stride=128):
    """Keep native NaNs and native-grid windows inside one Protocol recording/state."""
    subjects = list(range(101,109))
    if test_subject not in subjects:raise ValueError('PAMAP2 test subject is 101..108')
    validation_subject = subjects[(subjects.index(test_subject)+1)%len(subjects)]
    # UCI 54-column schema, zero based: IMU blocks start at 3,20,37.
    columns = [start+offset for start in (3,20,37) for offset in (1,2,3,7,8,9)]
    x,y,group,split = [],[],[],[]
    for subject in subjects:
        table = np.loadtxt(root/'Protocol'/f'subject{subject}.dat', dtype=np.float32, ndmin=2)
        for start in range(0,len(table)-window+1,stride):
            clip=table[start:start+window]
            activity=int(clip[0,1])
            if activity not in PAMAP_ACTIVITIES or not np.all(clip[:,1]==activity):continue
            if not np.all(np.diff(clip[:,0])>0) or np.max(np.diff(clip[:,0]))>.015:continue
            x.append(clip[:,columns]);y.append(PAMAP_ACTIVITIES.index(activity));group.append(subject)
            split.append(2 if subject==test_subject else 1 if subject==validation_subject else 0)
    if not x:raise ValueError('No valid PAMAP2 Protocol windows')
    data=dict(x=np.stack(x),y=np.asarray(y,np.int64),group=np.asarray(group),split=np.asarray(split))
    rng=np.random.default_rng(seed);keep=[]
    for subject in subjects:
        for label in range(len(PAMAP_ACTIVITIES)):
            ids=np.flatnonzero((data['group']==subject)&(data['y']==label))
            if subject not in (test_subject,validation_subject):ids=rng.choice(ids,min(cap,len(ids)),replace=False)
            keep.extend(ids)
    ids=np.sort(keep);data={key:value[ids] for key,value in data.items()}
    data['mask']=np.isfinite(data['x'])
    return data, [list(range(start,start+6)) for start in (0,6,12)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['har','pamap2'],required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-subject',type=int,default=101);parser.add_argument('--seed',type=int,default=13)
    args=parser.parse_args()
    data,groups=har(args.root,args.seed) if args.dataset=='har' else pamap2(args.root,args.test_subject,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**data)
    config=dict(channel_groups=groups,input_shape=list(data['x'].shape[1:]),classes=int(data['y'].max())+1)
    args.output.with_suffix('.config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    support={str(subject):{str(label):int(((data['group']==subject)&(data['y']==label)).sum()) for label in np.unique(data['y'])}
             for subject in np.unique(data['group'])}
    args.output.with_suffix('.support.json').write_text(json.dumps(support,indent=2),encoding='utf-8')


if __name__=='__main__':main()
