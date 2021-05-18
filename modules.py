import torch
import torch.nn as nn


# shape of input:[C, T, embed_size]
# shape of output:[C, T,embed_size]
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.query = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.key = nn.Linear(self.per_dim, self.per_dim, bias=False)
        self.value = nn.Linear(self.per_dim, self.per_dim, bias=False)

    def forward(self, input):
        C, T, E = input.shape
        input = torch.reshape(input, [C, T, self.heads, self.per_dim])  # [C, T, heads, per_dim]

        # compute q, k, v
        queries = self.query(input)
        keys = self.key(input)
        values = self.value(input)

        # scaled Dot_Product and attention
        similarity_scores = torch.softmax(torch.einsum("cthd, cqhd->ctqh",
                                                       (queries, keys)) / (self.embed_size ** (1 / 2)), dim=0)  # [C, T, T, H]

        # print(similarity_scores.shape)
        attn = torch.einsum("ctth, ckhd->cthd", (similarity_scores, values))  # [C, T, H, per_dim]
        # print(attn.shape)

        attn = torch.reshape(attn, [C, T, self.embed_size])

        return attn


# shape of input:[C, T, embed_size]
# shape of output:[C, T, embed_size]
class TimeTransformerNetBlock(nn.Module):
    def __init__(self, embed_size, heads, map_dim):
        super(TimeTransformerNetBlock, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.fc1 = nn.Linear(embed_size, map_dim)
        self.fc2 = nn.Linear(map_dim, embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, input):

        attn = self.attention(input)  # [C, T, embed_size]

        x = input + attn
        norm1 = self.norm1(x)
        out = self.fc2(self.fc1(x + norm1))
        out = self.norm2(out)

        return out


