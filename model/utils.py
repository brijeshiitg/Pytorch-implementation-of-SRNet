"""This module provide building blocks for SRNet."""

from torch import nn
from torch import Tensor


class ConvBn(nn.Module):
    """Provides utility to create different types of layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Constructor.
        Args:
            in_channels (int): no. of input channels.
            out_channels (int): no. of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns Conv2d followed by BatchNorm.

        Returns:
            Tensor: Output of Conv2D -> BN.
        """
        return self.batch_norm(self.conv(inp))


class Type1(nn.Module):
    """Creates type 1 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convbn = ConvBn(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 1 layer of SRNet.
        Args:
            inp (Tensor): input tensor.
        Returns:
            Tensor: Output of type 1 layer.
        """
        return self.relu(self.convbn(inp))


class Type2(nn.Module):
    """Creates type 2 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(in_channels, out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 2 layer of SRNet.
        Args:
            inp (Tensor): input tensor.
        Returns:
            Tensor: Output of type 2 layer.
        """
        return inp + self.convbn(self.type1(inp))


class Type3(nn.Module):
    """Creates type 3 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 3 layer of SRNet.
        Args:
            inp (Tensor): input tensor.

        Returns:
            Tensor: Output of type 3 layer.
        """
        out = self.batch_norm(self.conv1(inp))
        out1 = self.pool(self.convbn(self.type1(inp)))
        return out + out1


class Type4(nn.Module):
    """Creates type 4 layer of SRNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns type 4 layer of SRNet.
        Args:
            inp (Tensor): input tensor.
        Returns:
            Tensor: Output of type 4 layer.
        """
        return self.gap(self.convbn(self.type1(inp)))


if __name__ == "__main__":
    import torch

    tensor = torch.randn((1, 1, 256, 256))
    lt1 = Type1(1, 64)
    output = lt1(tensor)
    print(output.shape)
