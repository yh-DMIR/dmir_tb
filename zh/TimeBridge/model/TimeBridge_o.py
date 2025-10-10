import torch
import torch.nn as nn
from layers.Embed import PatchEmbed
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.Transformer_EncDec import TSEncoder, IntAttention, PatchSampling, CointAttention


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin  # long-term with temporal

        self.ia_layers = configs.ia_layers
        self.pd_layers = configs.pd_layers
        self.ca_layers = configs.ca_layers
        self.sum = self.ia_layers + self.pd_layers + self.ca_layers
        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        self.embedding1 = PatchEmbed(configs, num_p=self.num_p)
        self.embedding2 = PatchEmbed(configs, num_p=self.num_p)

        layers = self.layers_init(configs)
        self.encoder1 = TSEncoder(layers)
        self.encoder2 = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.num_p * configs.d_model, configs.pred_len, bias=False)
        )

        self.decoder2 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )

    def layers_init(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for i in range(configs.ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for i in range(configs.ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        x_enc1 = self.embedding1(x_enc, x_mark_enc)

        x_enc2 = self.embedding2(x_enc, x_mark_enc)

        # 分支
        enc_out1 = self.encoder1(x_enc1)[0][self.ia_layers-1][:, :self.c_in, ...]

        #预测y
        enc_out2 = self.encoder2(x_enc2)[0][-1][:, :self.c_in, ...]

        dec_out1 = self.decoder1(enc_out1).transpose(-1, -2)

        dec_out2 = self.decoder2(enc_out2).transpose(-1, -2)

        dec_out = dec_out1 + dec_out2

        return dec_out * std + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
