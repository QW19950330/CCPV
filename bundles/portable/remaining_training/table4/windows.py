"""Exact old train Dataset segment; no model import or payload read."""
import torch
from torch.utils.data import Dataset
AUGMENTATION_SEED=2026102418
def bind_api(api):
 global mask_history
 mask_history=api.mask_history

class Windows(Dataset):
    def __init__(self,arrays):self.a=arrays;self.indices=arrays['train_indices'];self.epoch=0
    def __len__(self):return len(self.indices)
    def __getitem__(self,item):
        ix=int(self.indices[item]);a=self.a
        h=torch.tensor(a['states'][ix,:10],dtype=torch.float32);v=torch.tensor(a['valid'][ix,:10],dtype=torch.bool)
        teams=torch.tensor(a['teams'] if a['teams'].ndim==1 else a['teams'][ix],dtype=torch.long)
        clean,usable,_=mask_history(h[None],v[None],teams,[int(a['source_indices'][ix])],
            condition='tailgap2',seed=AUGMENTATION_SEED,epoch=self.epoch)
        return dict(history=clean[0],valid=usable[0],history_seen=v.any(0),teams=teams,
            target=torch.tensor(a['states'][ix,10:30],dtype=torch.float32),
            target_valid=torch.tensor(a['valid'][ix,10:30],dtype=torch.bool),
            position=torch.tensor(a['position_targets'][ix,10:30],dtype=torch.float32),
            position_valid=torch.tensor(a['position_valid'][ix,10:30],dtype=torch.bool),
            item_index=torch.tensor(item,dtype=torch.long))
