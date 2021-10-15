import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        f1 = 32
        f2 = 64

        self.feature_layer = nn.Sequential(
            torch.nn.Conv2d(1, f1, kernel_size=5, stride=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(f1, f2, kernel_size=5, stride=3),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Linear(256, 10)
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.out_layer(x)
        return x


if __name__ == '__main__':
    image = torch.randn([2, 1, 25, 50])
    model = ConvNet()

    output = model(image)
    print(output.size())