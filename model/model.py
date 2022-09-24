""" This module creates SRNet model."""
import torch
from torch import nn
from model.utils import Type1, Type2, Type3, Type4


class Srnet(nn.Module):
    """This is SRNet model class."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self.type1s = nn.Sequential(Type1(1, 64), Type1(64, 16))
        self.type2s = nn.Sequential(
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
        )
        self.type3s = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        self.type4 = Type4(256, 512)
        self.dense = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp):
        """Returns logits for input images.
        Args:
            inp (Tensor): input image tensor of shape (Batch, 1, 256, 256)
        Returns:
            Tensor: Logits of shape (Batch, 2)
        """
        out = self.type1s(inp)
        out = self.type2s(out)
        out = self.type3s(out)
        out = self.type4(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        return self.softmax(out)


if __name__ == "__main__":
    image = torch.randn((1, 1, 256, 256))
    net = Srnet()
    print(net(image).shape)
