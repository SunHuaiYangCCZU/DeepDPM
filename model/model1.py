import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BioWaveKAN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.scale = nn.Parameter(torch.normal(1.0, 0.1, (1, in_dim)))
        self.translate = nn.Parameter(torch.zeros(1, in_dim))

        self.wave_weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.base_weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        nn.init.kaiming_normal_(self.wave_weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.orthogonal_(self.base_weight)

        self.bn = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(0.1)

    def _bio_wavelet(self, x):
        x = x.unsqueeze(1)
        x = (x - self.translate) / self.scale.clamp(min=1e-3)
        wavelet = torch.cos(3 * x) * torch.exp(-0.5 * x ** 2)
        return math.pi ** (-0.25) * wavelet

    def forward(self, x):
        wavelet = self._bio_wavelet(x)
        wave_out = torch.einsum('boi,oi->bo', wavelet, self.wave_weight)

        base_out = F.linear(x, self.base_weight)

        return self.drop(self.bn(wave_out + 0.3 * base_out))


class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding != 0 else x


class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super().__init__()
        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0
        self.blocks = nn.ModuleList([
            nn.Linear(in_features // num_blocks, out_features // num_blocks)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = x.chunk(len(self.blocks), dim=-1)
        x = [b(xi) for b, xi in zip(self.blocks, x)]
        return torch.cat(x, dim=-1)


class mLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)

        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)

        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)

        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state

        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)

        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)

        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)

        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * (v * k)
        n_t = f * n_prev + i * k
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0]

        output = self.group_norm(h_t) + x_skip
        output = output * F.silu(x_up_right)
        return self.down_proj(output) + x, (h_t, c_t, n_t, m_t)


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers=1, batch_first=True):
        super().__init__()
        self.layers = nn.ModuleList([
            mLSTMBlock(
                input_size if i == 0 else hidden_size,
                hidden_size,
                num_heads
            ) for i in range(num_layers)
        ])
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()
        device = x.device

        state = [
            (
                torch.zeros(batch_size, self.layers[0].hidden_size, device=device),
                torch.zeros(batch_size, self.layers[0].hidden_size, device=device),
                torch.zeros(batch_size, self.layers[0].hidden_size, device=device),
                torch.zeros(batch_size, self.layers[0].hidden_size, device=device)
            )
            for _ in range(len(self.layers))
        ]

        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            new_state = []
            for layer_idx, layer in enumerate(self.layers):
       
                h_prev, c_prev, n_prev, m_prev = state[layer_idx]

                x_t, layer_state = layer(x_t, (h_prev, c_prev, n_prev, m_prev))
                new_state.append(layer_state)

            state = new_state
            outputs.append(x_t)

        output = torch.stack(outputs)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, state

class MoRFPredictionBranch1(nn.Module):
    def __init__(self, feat_dim=1024, latent_dim=512, num_layers=2, max_seq_len=300):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            BioWaveKAN(feat_dim, 768),
            #nn.Linear(feat_dim, 768),  
            nn.GELU(),
            nn.LayerNorm(768),
            BioWaveKAN(768, latent_dim)
            #nn.Linear(768, latent_dim)  
        )

        self.mlstm = mLSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_heads=8,
            num_layers=num_layers,
            batch_first=True
        )
        #self.sequence_modeler = nn.Identity()

        self.pos_encoder = nn.Embedding(max_seq_len, latent_dim)
        self.pad_value = 0

        self.feature_decoder = nn.Sequential(
            BioWaveKAN(latent_dim, 768),
            #nn.Linear(latent_dim, 768), 
            nn.GELU(),
            nn.LayerNorm(768),
            BioWaveKAN(768, 512)
            #nn.Linear(768, 512) 
        )

    def forward(self, x, lengths=None):
        B, T, D = x.size()

        if lengths is not None:
            mask = (torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)).clone().detach()
        else:
            mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        encoded = self.feature_encoder(x.view(B * T, D)).view(B, T, -1)

        positions = torch.arange(T, device=x.device).expand(B, T)
        pos_emb = self.pos_encoder(positions)
        encoded = encoded + pos_emb

        encoded_masked = encoded.masked_fill(mask.unsqueeze(-1), self.pad_value)

        lstm_out, _ = self.mlstm(encoded_masked)
        lstm_out_masked = lstm_out.masked_fill(mask.unsqueeze(-1), self.pad_value)

        # sequence_out = self.sequence_modeler(encoded_masked)
        # sequence_out_masked = sequence_out.masked_fill(mask.unsqueeze(-1), self.pad_value)

        decoded = self.feature_decoder(lstm_out_masked.view(B * T, -1)).view(B, T, -1)
        #decoded = self.feature_decoder(sequence_out_masked.view(B * T, -1)).view(B, T, -1)
        #decoded = self.feature_decoder(encoded_masked.view(B * T, -1)).view(B, T, -1)

        return decoded
