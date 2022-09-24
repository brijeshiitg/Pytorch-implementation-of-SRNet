"""This module provide the data sample for training."""

import os
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset

import imageio as io

from opts.options import arguments

opt = arguments()
# pylint: disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101


class DatasetLoad(Dataset):
    """This class returns the data samples."""

    def __init__(
        self,
        cover_path: str,
        stego_path: str,
        size: int,
        transform: Tuple = None,
    ) -> None:
        """Constructor.

        Args:
            cover_path (str): path to cover images.
            stego_path (str): path to stego images.
            size (int): no. of images in any of (cover / stego) directory for
              training.
            transform (Tuple, optional): _description_. Defaults to None.
        """
        self.cover = cover_path
        self.stego = stego_path
        self.transforms = transform
        self.data_size = size

    def __len__(self) -> int:
        """returns the length of the dataset."""
        return self.data_size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Returns the (cover, stego) pairs for training.

        Args:
            index (int): a random int value in range (0, len(dataset)).
        Returns:
            Tuple[Tensor, Tensor]: cover and stego pair.
        """
        index += 1
        img_name = str(index) + ".pgm"
        cover_img = io.imread(os.path.join(self.cover, img_name))
        stego_img = io.imread(os.path.join(self.stego, img_name))
        # pylint: disable=E1101
        label1 = torch.tensor(0, dtype=torch.long).to(device)
        label2 = torch.tensor(1, dtype=torch.long).to(device)
        # pylint: enable=E1101
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)
            sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [label1, label2]
        return sample
