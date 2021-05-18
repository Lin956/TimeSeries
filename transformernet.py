import torch
import torch.nn as nn
from modules import TimeTransformerNetBlock


# shape of input:[C, T, 1]
# shape of output:[C, out_T_dim, 1]
class TransformerNet(nn.Module):
    def __init__(self, T, embed_size, out_T_dim, heads, num_layers, map_dim):
        super(TransformerNet, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.out_T_dim = out_T_dim
        self.num_layers = num_layers
        # 维数变换
        self.conv1 = nn.Conv2d(1, embed_size, 1)
        """
        self-attention
        add & normal
        Linear
        add & normal
        """
        self.transformerblock = TimeTransformerNetBlock(embed_size, heads, map_dim)
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
        self.conv3 = nn.Conv2d(T, out_T_dim, 1)

    def forward(self, input):
        C, T, D = input.shape

        input.unsqueeze_(0)  # [1, C, T, 1]
        input = input.permute(0, 3, 2, 1)  # [1, 1, T, C]
        input = self.conv1(input)  # [1, embed_size, T, C]
        input = input.permute(0, 3, 2, 1)  # [1, C, T, embed_size]
        input.squeeze_(0)

        for i in range(self.num_layers):
            tran_out = self.transformerblock(input)
            input = tran_out  # [C, T, embed_size]

        # output:[C, out_T_dim, 1]
        tran_out.unsqueeze_(0)
        tran_out = tran_out.permute(0, 3, 2, 1)
        out = self.conv2(tran_out)  # [1, 1, T, C]
        out = out. permute(0, 2, 1, 3)  # [1, T, 1, C]
        predict_out = self.conv3(out)  # [1, out_T_dim, 1, C]
        predict_out = predict_out.permute(0, 3, 1, 2)  # [1, C, out_T_dim, 1]
        predict_out.squeeze_(0)

        return predict_out
