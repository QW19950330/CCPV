"""Exact original optimizer and epoch-one batch construction fragments."""
def optimizer_for(model):
 import torch
 return torch.optim.Adam(model.parameters(),lr=.0003,weight_decay=0.)

def batch_to(raw,device):return {k:v.to(device) for k,v in raw.items() if k!='item_index'}

def first_batch(api,a,seed):
 import torch,windows
 from torch.utils.data import DataLoader
 windows.bind_api(api);dataset=windows.Windows(a);dataset.epoch=1;gen=torch.Generator().manual_seed(100003003+seed+7)
 return next(iter(DataLoader(dataset,batch_size=32,shuffle=True,generator=gen,num_workers=0)))
