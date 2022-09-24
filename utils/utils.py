"""This module provides utility function for training."""
import os
import re
from typing import Any, Dict
import torch
from torch import nn

from opts.options import arguments

opt = arguments()


def saver(state: Dict[str, float], save_dir: str, epoch: int) -> None:
    torch.save(state, save_dir + "net_" + str(epoch) + ".pt")


def latest_checkpoint() -> int:
    """Returns latest checkpoint."""
    if os.path.exists(opt.checkpoints_dir):
        all_chkpts = "".join(os.listdir(opt.checkpoints_dir))
        if len(all_chkpts) > 0:
            latest = max(map(int, re.findall("\d+", all_chkpts)))
        else:
            latest = None
    else:
        latest = None
    return latest


def adjust_learning_rate(optimizer: Any, epoch: int) -> None:
    """Sets the learning rate to the initial learning_rate and decays by 10
    every 30 epochs."""
    learning_rate = opt.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


# Weight initialization for conv layers and fc layers
def weights_init(param: Any) -> None:
    """Initializes weights of Conv and fully connected."""

    if isinstance(param, nn.Conv2d):
        torch.nn.init.xavier_uniform_(param.weight.data)
        if param.bias is not None:
            torch.nn.init.constant_(param.bias.data, 0.2)
    elif isinstance(param, nn.Linear):
        torch.nn.init.normal_(param.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(param.bias.data, 0.0)
