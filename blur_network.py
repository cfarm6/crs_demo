import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path

from collections import OrderedDict
import torchvision
from kornia.filters import *


class GaussianBlur(nn.Module):
    def __init__(self, shape: Tuple[int, int, int, int], dtype=torch.float):
        super().__init__()
        # self.shape = shape
        # self.dtype = dtype
        # kernel = torch.tensor([
        #     [1, 4, 7, 4, 1],
        #     [4, 16, 26, 16, 4],
        #     [7, 26, 41, 26, 7],
        #     [4, 16, 26, 16, 4],
        #     [1, 4, 7, 4, 1],

        # ], dtype=torch.float)
        # self.kernel = kernel / 273
        # # Define Layers
        # _, _, _, c = self.shape
        # self.blur = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=1,
        #     padding=2,
        #     kernel_size=5,
        #     bias=False
        # )
        # self.blur.requires_grad_(False)
        # # Initialize Weights
        # self.blur.weight[:, :, :, :] = self.kernel

    def forward(self, x):
        # b = x[0, 0, :, :]
        # g = x[0, 1, :, :]
        # r = x[0, 2, :, :]

        # b_ = b.type(torch.float)
        # b_ = torch.unsqueeze(b_, dim=0)
        # b_ = torch.unsqueeze(b_, dim=0)

        # g_ = g.type(torch.float)
        # g_ = torch.unsqueeze(g_, dim=0)
        # g_ = torch.unsqueeze(g_, dim=0)

        # r_ = r.type(torch.float)
        # r_ = torch.unsqueeze(r_, dim=0)
        # r_ = torch.unsqueeze(r_, dim=0)

        # b_ = self.blur(b_)[0][0]
        # g_ = self.blur(g_)[0][0]
        # r_ = self.blur(r_)[0][0]

        # output = torch.stack([b_, g_, r_], dim=-1)
        # output = torch.unsqueeze(output, dim=0)
        return box_blur(x, (3, 3))
def export():
    output_dir = Path(__file__).parent / 'out'
    output_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(output_dir=output_dir)
    print('Done.')


def export_onnx(output_dir):
    """
    Exports the model to an ONNX file.
    """
    # Define the expected input shape (dummy input)
    shape = (1, 3, 300, 300)
    # Create the Model
    model = GaussianBlur(shape=shape, dtype=torch.float)
    X = torch.ones(shape, dtype=torch.float)
    torch.onnx.export(
        model,
        X,
        f'{output_dir.as_posix()}/model.onnx',
        opset_version=11,
        do_constant_folding=False
    )

export()
