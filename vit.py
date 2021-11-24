'''
Author: your name
Date: 2021-11-06 04:45:03
LastEditTime: 2021-11-19 02:00:08
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /renet/vit-pytorch/test_vit.py
'''
import torch
from models.others.vit_test import ViT
from models.others.sce import SpatialContextEncoder

v = ViT(
    image_size = 84,
    patch_size = 4,
    num_classes = 100,
    dim = 64,#(H*W*C)
    depth = 6,
    heads = 16,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 84, 84)
print(img.size())
preds = v(img) # (1, 1000)
print(preds.size())
