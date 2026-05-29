import torch
from torch import nn
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DBLNEmbed
from mamba_ssm import Mamba

from layers.DBLN_EncDec import Encoder, EncoderLayer, HVMamba

class Model(nn.Module):
    def __init__(self, configs):

        super().__init__()

        self.decomp = configs.decomp
        # patching and embedding
        self.patch_embedding = DBLNEmbed(self.decomp, configs.seq_len, configs.enc_in,
                                              configs.patch_len, configs.d_model, configs.ls_scale)

        # positional embedding
        num_patches_long = self.patch_embedding.num_patches_long
        self.pos_embed_long = nn.Parameter(torch.zeros(1, num_patches_long, configs.d_model), requires_grad=True)
        if self.decomp:
            num_patches_short = self.patch_embedding.num_patches_short
            self.pos_embed_short = nn.Parameter(torch.zeros(1, num_patches_short, configs.d_model), requires_grad=True)
        self.pos_drop = nn.Dropout(p=configs.dropout)
        if configs.hvmamba:
            extractor_l = HVMamba(d_model = configs.d_model, seqlen=num_patches_long, d_state = configs.d_state, d_conv = configs.d_conv, expand = configs.expand)
            extractor_s = HVMamba(d_model = configs.d_model, seqlen=num_patches_short, d_state = configs.d_state, d_conv = configs.d_conv, expand = configs.expand) if configs.decomp else None
        else:
            extractor_l = Mamba(d_model = configs.d_model, d_state = configs.d_state, d_conv = configs.d_conv, expand = configs.expand)
            extractor_s = Mamba(d_model = configs.d_model, d_state = configs.d_state, d_conv = configs.d_conv, expand = configs.expand) if configs.decomp else None
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    extractor_l=extractor_l,
                    extractor_s=extractor_s,
                    interactor=AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=True), configs.d_model, configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    decomp = configs.decomp,
                    fuse = configs.fuse,
                    pre_norm = configs.pre_norm
                ) for l in range(configs.e_layers)
            ],
            decomp = configs.decomp,
            norm_layer = nn.LayerNorm(configs.d_model)
        )

        # Head
        self.projection = nn.Linear(configs.d_model, configs.num_class)  

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec): # [B,L,D]
        x = x_enc.permute(0, 2, 1)

        if self.decomp:
            # patch embedding
            x_long, x_short = self.patch_embedding(x)                   # patch embedding (B, L, D)
           
            # positional embedding
            x_long = self.pos_drop(x_long + self.pos_embed_long)        # posit embedding (B, 2N, D)
            x_short = self.pos_drop(x_short + self.pos_embed_short)     # posit embedding (B, N, D)  

            # Encoder
            x_long, x_short, attns_long, attns_short = self.encoder(x_long, x_short)

            # cat
            x_out = torch.cat((x_long, x_short), dim=1)


        else:
            x_out = self.patch_embedding(x)
            x_out = self.pos_drop(x_out + self.pos_embed_long)
            x_short = None  
            x_out, attns_out = self.encoder(x_out, x_short)

        # Decoder 
        x_out = x_out.mean(dim=1)
        output = self.projection(x_out)  # (batch_size, num_classes)

        return output
    


