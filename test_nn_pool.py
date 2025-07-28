import torch

input = torch.tensor([[1, 2, 0, 3, 1], [0, 1, 2, 3, 1], [1, 2, 1, 0, 0], [5, 2, 3, 1, 1], [2, 1, 0, 1, 1]])

input = torch.reshape(input,(-1,1,5,5))

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Pool = torch.nn.MaxPool2d(3,ceil_mode=True)

    def forward(self,x):
        output = self.Pool(x)
        return output

mod = Model()
print(mod(input))
