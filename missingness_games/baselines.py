"""Classification adaptations using the actual BSD-licensed PyPOTS backbones."""
import torch
from torch import nn

from .vendor.pypots.nn.modules.grud.backbone import BackboneGRUD
from .vendor.pypots.nn.modules.brits.backbone import BackboneBRITS
from .vendor.pypots.nn.modules.saits.backbone import BackboneSAITS


def temporal_inputs(values,mask):
    gap=torch.zeros_like(values[:,0]);last=torch.zeros_like(gap);deltas=[];locf=[]
    for step in range(values.shape[1]):
        gap=1+(1-mask[:,step-1])*gap if step else torch.zeros_like(gap)
        deltas.append(gap);locf.append(last)
        last=torch.where(mask[:,step].bool(),values[:,step],last)
    return torch.stack(deltas,1),torch.stack(locf,1)


class ClassificationBackbone(nn.Module):
    """Retain GRU-D decay, BRITS bidirectionality or SAITS DMSA and reconstruction."""
    def __init__(self,name,steps,features,width,classes,config):
        super().__init__();self.name=name;self.width=width
        if name=='grud':self.backbone=BackboneGRUD(steps,features,width)
        elif name=='brits':self.backbone=BackboneBRITS(steps,features,width)
        elif name=='saits':self.backbone=BackboneSAITS(steps,features,2,width,2,width//2,width//2,2*width,0.,0.)
        elif name=='timesnet':
            from types import SimpleNamespace
            from .vendor.timesnet.TimesNet import Model
            self.backbone=Model(SimpleNamespace(task_name='classification',seq_len=steps,label_len=0,pred_len=0,
                top_k=min(3,steps//2),d_model=width,d_ff=2*width,num_kernels=3,e_layers=2,
                enc_in=3*features,embed='timeF',freq='h',dropout=0.,num_class=classes))
        elif name=='biot':
            from .vendor.biot.biot import BIOTClassifier
            fft=int(config.get('biot_n_fft',min(64,steps)))
            self.backbone=BIOTClassifier(emb_size=width,heads=2,depth=2,n_classes=classes,n_channels=3*features,n_fft=fft,hop_length=max(1,fft//2))
        else:raise ValueError('Unknown external backbone')
        self.projection=nn.Linear(3*features,width) if name=='saits' else nn.Identity()
        self.auxiliary_loss=None

    def forward(self,values,mask):
        delta,locf=temporal_inputs(values,mask)
        if self.name in ('timesnet','biot'):
            self.auxiliary_loss=values.new_zeros(())
            augmented=torch.cat([values,mask,delta/values.shape[1]],-1)
            if self.name=='timesnet':return self.backbone(augmented,mask.any(-1).float(),None,None)
            return self.backbone(augmented.transpose(1,2))
        if self.name=='grud':
            _,hidden=self.backbone(values,mask,delta,torch.zeros_like(values[:,0]),locf)
            self.auxiliary_loss=hidden.new_zeros(())
            return hidden
        if self.name=='brits':
            reverse_values=values.flip(1);reverse_mask=mask.flip(1)
            reverse_delta,_=temporal_inputs(reverse_values,reverse_mask)
            result=self.backbone(dict(forward=dict(X=values,missing_mask=mask,deltas=delta),backward=dict(X=reverse_values,missing_mask=reverse_mask,deltas=reverse_delta)))
            self.auxiliary_loss=result[5]+result[6]
            return (result[3]+result[4])/2
        result=self.backbone(values,mask)
        self.auxiliary_loss=sum(((estimate-values).abs()*mask).sum()/mask.sum().clamp_min(1) for estimate in result[:3])/3
        imputed=mask*values+(1-mask)*result[2]
        representation=torch.cat([imputed,mask,delta/values.shape[1]],-1).mean(1)
        return self.projection(representation)
