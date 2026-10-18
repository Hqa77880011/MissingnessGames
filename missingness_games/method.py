import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;d=c.get('input_shape',[24,6])[-1];w=c.get('width',32)
        self.gru=nn.GRU(2*d,w,batch_first=True);self.head=nn.Linear(w,c.get('classes',3));self.register_buffer('memory',torch.zeros(4))
    def policy(self,x,index,observed=None):
        observed=torch.ones_like(x,dtype=torch.bool) if observed is None else observed.bool()
        n,time,d=x.shape;gap=torch.zeros(n,d,device=x.device);history=torch.zeros_like(gap);budget=torch.full((n,),int(time*d*self.c.get('deletion_fraction',.25)),device=x.device);masks=[]
        for t in range(time):
            if index==0:priority=torch.rand_like(gap)
            elif index==1:priority=(torch.arange(d,device=x.device)[None]==(t//4)%d).float().expand(n,-1)
            elif index==2:priority=abs(history)
            elif index==3:priority=torch.cos(torch.arange(d,device=x.device)[None]+t/3).expand(n,-1)
            elif index==4:priority=-abs(history)
            elif index==5:priority=torch.sin(torch.arange(d,device=x.device)[None]*6.283185/max(d,1)+(t//max(1,self.c.get('max_gap',4)))*3.141593).expand(n,-1)
            else:raise ValueError('Unknown missingness policy')
            legal=(gap<self.c.get('max_gap',4))&observed[:,t];count=torch.minimum(budget,torch.full_like(budget,max(1,d//3)));delete=torch.zeros_like(gap,dtype=torch.bool)
            for row in range(n):
                order=priority[row].masked_fill(~legal[row],-torch.inf).argsort(descending=True);chosen=order[:int(count[row])];chosen=chosen[legal[row,chosen]];delete[row,chosen]=True
            mask=((~delete)&observed[:,t]).float();budget-=delete.sum(1);gap=(gap+1)*delete;history=.9*history+.1*x[:,t].detach()*mask;masks.append(mask)
        return torch.stack(masks,1)
    def classify(self,x,mask):
        _,state=self.gru(torch.cat([x*mask,mask],-1));return self.head(state[-1])
    def forward(self,b):
        raw=b['x'].float();observed=b.get('mask',torch.isfinite(raw)).bool();x=torch.nan_to_num(raw)
        clean=self.classify(x,observed.float())
        if not self.training:return {'probability':clean.softmax(-1),'clean_logits':clean}
        logits=torch.stack([self.classify(x,self.policy(x,k,observed)) for k in range(4)],1)
        probability=logits.softmax(-1).mean(1)
        if self.c['variant']=='baseline':probability=clean.softmax(-1)
        return {'probability':probability,'policy_logits':logits,'clean_logits':clean}
    def selection_metrics(self,b):
        """Frozen clean eligibility and worst represented-policy validation loss."""
        raw=b['x'].float();observed=b.get('mask',torch.isfinite(raw)).bool();x=torch.nan_to_num(raw);y=b['y'].long()
        with torch.no_grad():
            clean=self.classify(x,observed.float())
            losses=torch.stack([F.cross_entropy(self.classify(x,self.policy(x,k,observed)),y,reduction='none') for k in range(4)],1)
        return int((clean.argmax(-1)!=y).sum()),float(losses.max(1).values.sum())
    def objective(self,b,p):
        y=b['y'].long();clean=F.cross_entropy(p['clean_logits'],y)
        if self.c['variant']=='baseline' or 'policy_logits' not in p:return clean
        losses=torch.stack([F.cross_entropy(p['policy_logits'][:,k],y) for k in range(4)])
        beta=0. if self.c['variant']=='no_memory' else self.c.get('memory',.9)
        if self.training:self.memory.mul_(beta).add_((1-beta)*losses.detach())
        weight=(self.memory/self.c.get('temperature',.2)).softmax(0).detach()
        if self.c['variant']=='fixed_bank':weight=torch.ones_like(weight)/4
        if self.c['variant']=='random_only':weight=F.one_hot(torch.tensor(0,device=y.device),4).float()
        return clean+(weight*losses).sum()


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,24,6)).astype('float32');y=np.argmax(x.mean(1)[:,:3],1)
    return {'x':x,'y':y,'group':np.arange(n)}
