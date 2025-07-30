import torch


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=2, stride=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, padding=2, stride=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, padding=2, stride=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return  x

if __name__ == '__main__':
    mod = Model()
    input = torch.ones([64, 3, 32, 32])
    output = mod(input)
    print(output.shape)