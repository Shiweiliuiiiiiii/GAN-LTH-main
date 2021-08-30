import torch
EMA = torch.tensor([[1,1,0],[0,1,0]])
new_mask = torch.tensor([[1,0,1],[0,1,0]])
i = torch.eye(2,3)
print(i)
a = ((EMA).data.byte() ^ new_mask.data.byte()) & new_mask.data.byte()
print(i.add_(0.999*a))