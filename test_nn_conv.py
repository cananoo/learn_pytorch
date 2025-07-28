import torch

input = torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]])
weight = torch.tensor([[1,2,1],[0,1,0],[2,1,0]])

input = torch.reshape(input,(1,1,5,5))
weight = torch.reshape(weight,(1,1,3,3))

output = torch.nn.functional.conv2d(input, weight,stride=1)

print(output)