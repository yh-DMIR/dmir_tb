import torch
import torch.nn as nn
from layers.Embed import PatchEmbed
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.Transformer_EncDec import TSEncoder, IntAttention, PatchSampling, CointAttention


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        self.embedding1 = PatchEmbed(configs, num_p=self.num_p)
        self.embedding2 = PatchEmbed(configs, num_p=self.num_p)


        layer2 = self.layers_init2(configs)
        layer3 = self.layers_init3(configs)
        layer4 = self.layers_init4(configs)

        self.encoder21 = TSEncoder(layer2)
        self.encoder22 = TSEncoder(layer2)
        self.encoder31 = TSEncoder(layer3)
        self.encoder32 = TSEncoder(layer3)
        self.encoder41 = TSEncoder(layer4)
        self.encoder42 = TSEncoder(layer4)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoderzx_mean = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        self.decoderzx_std = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        self.decoderzy_mean = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        self.decoderzy_std = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        mlp_hidden_dim = 128
        self.decoderzx = nn.Sequential(
            nn.Linear(self.c_in, mlp_hidden_dim),
            # nn.GELU(),  # 使用 GELU 作为非线性激活函数
            # nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),  # 使用 GELU 作为非线性激活函数
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),  # 使用 GELU 作为非线性激活函数
            nn.Linear(mlp_hidden_dim, self.c_in)
        )
        self.decoderzy = nn.Sequential(
            nn.Linear(self.c_in, mlp_hidden_dim),
            # nn.GELU(),  # 使用 GELU 作为非线性激活函数
            # nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),  # 使用 GELU 作为非线性激活函数
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),  # 使用 GELU 作为非线性激活函数
            nn.Linear(mlp_hidden_dim, self.c_in)
        )
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def layers_init2(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for i in range(configs.ia_layers)]

        return [*integrated_attention]


    def layers_init3(self, configs):
        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        return [*patch_sampling]

    def layers_init4(self, configs):
        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for i in range(configs.ca_layers)]

        return [*cointegrated_attention]


    def encoder_mean1(self, x_enc, x_mark_enc):
        x_mean = self.embedding1(x_enc, x_mark_enc)
        x_mean = self.encoder21(x_mean)[0][:, :self.c_in, ...]
        zx_mean = self.decoderzx_mean(x_mean).transpose(-1, -2)
        return zx_mean

    def encoder_std1(self, x_enc, x_mark_enc):
        x_std = self.embedding2(x_enc, x_mark_enc)
        x_std = self.encoder21(x_std)[0][:, :self.c_in, ...]
        zx_std = self.decoderzx_std(x_std).transpose(-1, -2)
        return zx_std

    def encoder_mean2(self, zx_mean):
        x_mean = self.encoder31(zx_mean)[0]
        x_mean = self.encoder41(x_mean)[0]
        zx_mean = self.decoderzy_mean(x_mean).transpose(-1, -2)
        return zx_mean

    def encoder_std2(self, zx_std):
        x_std = self.encoder32(zx_std)[0]
        x_std = self.encoder42(x_std)[0]
        zx_std = self.decoderzy_std(x_std).transpose(-1, -2)
        return zx_std

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        # x_enc = self.embedding(x_enc, x_mark_enc)
        # enc_out = self.encoder(x_enc)[0][:, :self.c_in, ...]
        # dec_out = self.decoder(enc_out).transpose(-1, -2)

        zx_mean = self.encoder_mean1(x_enc, x_mark_enc)
        zx_std = self.encoder_std1(x_enc, x_mark_enc)
        z_mean = self.decoderzx_mean(zx_mean)
        z_mean = z_mean.transpose(-1, -2)
        z_std = self.decoderzy_std(zx_std)
        z_std = z_std.transpose(-1, -2)
        x_out = self.reparametrize(z_mean, z_std)
        x_out = self.decoderzx(x_out)

        zy_mean = self.encoder_mean2(zx_mean)
        zy_std = self.encoder_std2(zx_std)
        y_out = self.reparametrize(zy_mean, zy_std)
        y_out = self.decoderzy(y_out)

        return y_out * std + mean, x_out * std + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]