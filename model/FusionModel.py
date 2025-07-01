import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class SEBlock(nn.Module):

    def __init__(self, d_model, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // reduction)
        self.fc2 = nn.Linear(d_model // reduction, d_model)

    def forward(self, x):
        se = torch.mean(x, dim=1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).unsqueeze(1)
        return x * se


class BiLCrossAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=16, dropout=0.1):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_kv = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.channel_attn = SEBlock(d_model)
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            Transpose(1, 2), 
            nn.LayerNorm(d_model),
            Transpose(1, 2), 
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        q = self.proj_q(q)
        kv = self.proj_kv(kv)

        attn_output, _ = self.attn(query=q, key=kv, value=kv)
        q = q + self.dropout(attn_output)
        q = self.channel_attn(q)

        combined = torch.cat([q, kv], dim=-1)
        combined = combined.transpose(1, 2)  # [batch_size, 2*d_model, seq_len]

        fused = self.fusion(combined)
        fused = fused.transpose(1, 2)  # [batch_size, seq_len, d_model]

        return fused


class MultiScaleSmoothing(nn.Module):
    def __init__(self, d_model, scales=(3, 5, 7)):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=s, padding=s // 2, bias=False)
            for s in scales
        ])
        for conv in self.convs:
            nn.init.constant_(conv.weight, 1 / conv.kernel_size[0])
            conv.weight.requires_grad = False

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, seq_len]

        smoothed = torch.stack([conv(x) for conv in self.convs], dim=0).mean(0)
        return smoothed.squeeze(1) 

class AttentionWeightedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x1, x2):
        attn1 = self.attention(x1)  # [batch_size, seq_len, 1]
        attn2 = self.attention(x2)  # [batch_size, seq_len, 1]

        weights = torch.cat([attn1, attn2], dim=-1)  # [batch_size, seq_len, 2]
        weights = F.softmax(weights, dim=-1)  # [batch_size, seq_len, 2]

        weight1 = weights[:, :, 0:1]  # [batch_size, seq_len, 1]
        weight2 = weights[:, :, 1:2]  # [batch_size, seq_len, 1]
        fused = weight1 * x1 + weight2 * x2

        return fused


class FusionModel(nn.Module):
    def __init__(self, d_model=512, num_heads=16, dropout=0.1):
        super().__init__()
        self.bil_cross1 = BiLCrossAttention(d_model, num_heads, dropout)
        self.bil_cross2 = BiLCrossAttention(d_model, num_heads, dropout)

        self.attention_fusion = AttentionWeightedFusion(d_model)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1)
        )

        self.smoothing = MultiScaleSmoothing(d_model)

    def forward(self, x1, x2):
        fused1 = self.bil_cross1(x1, x2)  # [batch_size, seq_len, d_model]
        fused2 = self.bil_cross2(x2, x1)  # [batch_size, seq_len, d_model]

        combined = self.attention_fusion(fused1, fused2)  # [batch_size, seq_len, d_model]

        raw_output = self.pred_head(combined)  # [batch_size, seq_len, 1]
        raw_output = raw_output.squeeze(-1)  # [batch_size, seq_len]

        smoothed = self.smoothing(raw_output)  # [batch_size, seq_len]

        return smoothed

