import torch
import torch.nn as nn
from torch.nn.functional import pad, softmax
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=3, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = x.unsqueeze(1)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class LV(nn.Module):
    def __init__(self, bands, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = 64
        self.conv1d_1 = nn.Conv1d(1, out_channels // 2, kernel_size=1)  # 1x1 Conv1d ->(1,32,46)
        self.conv1d_2 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)  # 1x3 Conv1d ->(1,32,46)
        self.conv1d_3 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=5, padding=2)  # 1x5 Conv1d ->(1,32,46)
        self.conv1d_4 = nn.Conv1d(1, out_channels // 2, kernel_size=1)  # 1x1 Conv1d after concatenation ->(1,32,46)

    def forward(self, x):
        #x = x.squeeze(1)  # Remove the singleton channel dimension
        x1, x2 = torch.chunk(x, 2, dim=2)
        x1 = self.conv1d_1(x1)
        x1 = self.conv1d_2(x1)
        x1 = self.conv1d_3(x1)
        x2 = self.conv1d_4(x2)
        x = torch.cat([x1, x2], dim=2)  # Concatenate along the channel dimension (1,64,46)
        #计算平均值，变成（1，1，46）
        x = x.mean(dim=1, keepdim=True)
        return x  # Add back the singleton channel dimension for consistency


class SpatialFrequencyFusionModule(nn.Module):
    def __init__(self, input_dim, output_channels=64):
        super(SpatialFrequencyFusionModule, self).__init__()
        self.input_dim = input_dim
        self.padded_dim = math.ceil(math.sqrt(input_dim)) ** 2

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(1, output_channels // 2, kernel_size=3, padding=1)  # First 2D conv
        self.conv2d_2 = nn.Conv2d(output_channels // 2, output_channels, kernel_size=3, padding=1)  # Second 2D conv
        self.conv2d_3 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)  # Third 2D conv

        # 2D to 1D conversion
        self.fc_out = nn.Linear(output_channels * int(math.sqrt(self.padded_dim)) ** 2, output_channels * input_dim)
        self.conv = nn.Conv1d(output_channels, 1, kernel_size=1)

    def forward(self, x):
        # Pad the input to the nearest square number
        batch_size, _, _ = x.size()
        pad_size = self.padded_dim - self.input_dim
        x = F.pad(x, (0, pad_size), 'constant', 0)

        # Convert 1D spectral data to 2D image
        x = x.view(batch_size, 1, int(math.sqrt(self.padded_dim)),
                   int(math.sqrt(self.padded_dim)))  # reshape to (batch_size, 1, sqrt(padded_dim), sqrt(padded_dim))

        # Apply 2D convolutions
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        #print(1,x.shape)

        # Flatten and convert back to 1D
        x = x.view(batch_size, -1)
        x = self.fc_out(x)
        x = x.view(batch_size, -1, self.input_dim)  # reshape to (batch_size, output_channels, input_dim)

        # Final 1D convolution
        x = self.conv(x)
        return x

class GLSL(nn.Module):
    def __init__(self, bands, hidden_dim=64, num_heads=3, mlp_dim=128, use_low_freq_branch=True):
        super().__init__()
        self.spectral_to_image_branch1 = SpatialFrequencyFusionModule(bands, output_channels=32)
        self.spectral_to_image_branch2 = SpatialFrequencyFusionModule(bands, output_channels=32)
        self.use_low_freq_branch = use_low_freq_branch
        self.layer_norm1 = nn.LayerNorm(bands)
        self.layer_norm2 = nn.LayerNorm(bands)
        self.w_msa1 = MultiHeadSelfAttention(embed_dim=bands, num_heads=num_heads)
        self.w_msa2 = MultiHeadSelfAttention(embed_dim=bands, num_heads=num_heads)
        self.layer_norm11 = nn.LayerNorm(bands)
        self.layer_norm22 = nn.LayerNorm(bands)
        self.mlp1 = nn.Sequential(
            nn.Linear(bands, mlp_dim),
            nn.GELU(),  # GELU activation
            nn.Dropout(0.9),
            nn.Linear(mlp_dim, bands)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(bands, mlp_dim),
            nn.GELU(),  # GELU activation
            nn.Dropout(0.9),
            nn.Linear(mlp_dim, bands)
        )

        self.fc_for_x1 = nn.Linear(bands, bands)
        self.fc_for_low_freq1 = nn.Linear(bands, bands)
        self.fc_for_x2 = nn.Linear(bands, bands)
        self.fc_for_low_freq2 = nn.Linear(bands, bands)

        self.low_freq_branch1 = LV(204)
        self.low_freq_branch2 = LV(204)

    def _initialize_weights(self):
        pass

        for name, param in self.w_msa.named_parameters():
            if 'weight' in name:
                xavier_uniform_(param)
            elif 'bias' in name:
                constant_(param, 0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)

        if self.use_low_freq_branch:
            self.low_freq_branch.apply(self._initialize_submodule_weights)  # 假设LowFrequencyBranch有相应的初始化逻辑
    def _initialize_submodule_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)

    def forward(self, x):
        #x = x.squeeze(1)  # Remove the channel dimension
        #第一层
        x1 = self.layer_norm1(x)
        x11 = self.w_msa1(x1)
        x1 = self.layer_norm11(x11)
        GV1 = x11 + self.mlp1(x1)
        LV1 = self.low_freq_branch1(x)
        x3 = self.spectral_to_image_branch1(x)

        GV1_mapped = self.fc_for_x1(GV1.squeeze(1))
        LV1_mapped = self.fc_for_low_freq1(LV1.squeeze(1))

        GV1_mapped = GV1_mapped.unsqueeze(1)
        LV1_mapped = LV1_mapped.unsqueeze(1)

        output1 = GV1_mapped + LV1_mapped + x3

        #第二层
        output1 = output1 + x
        output11 = self.layer_norm2(output1)
        output111 = self.w_msa2(output11)
        output11 = self.layer_norm22(output111)
        GV2 = output11 + self.mlp2(output11)
        LV2 = self.low_freq_branch2(output1)
        x33 = self.spectral_to_image_branch2(output1)

        GV2_mapped = self.fc_for_x2(GV2.squeeze(1))
        LV2_mapped = self.fc_for_low_freq2(LV2.squeeze(1))

        GV2_mapped = GV2_mapped.unsqueeze(1)
        LV2_mapped = LV2_mapped.unsqueeze(1)

        output2 = GV2_mapped + LV2_mapped + x33

        return output2  # Add back the channel dimension for consistency

class Network(nn.Module):
    def __init__(self, GLSL):
        super(Network, self).__init__()
        self.embedding_net = GLSL

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
