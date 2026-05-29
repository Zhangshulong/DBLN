import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, activation='gelu', pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm 
        self.w_1 = nn.Linear(d_in, d_hid)
        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.norm = nn.LayerNorm(d_in, eps=1e-6)
        
    def forward(self, x):
        res = x
        if self.pre_norm:
            x = self.norm(x)
        x = self.dropout(self.activation(self.w_1(x)))
        x = self.dropout(self.w_2(x))
        x = res + x
        if not self.pre_norm:
            x = self.norm(x)
        return x

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

class SSM_Path(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        #print("d_model", d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # SSM部分
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True
        # S4D x real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A) # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device)) # Keep in fp32
        self.D._no_weight_decay = True

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv-1,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B,D,L)
        Returns: same shape as hidden_states
        """
        _, _, seqlen = hidden_states.shape
        x = F.silu(self.conv1d(hidden_states)[..., :seqlen])

        A = -torch.exp(self.A_log.float()) 
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen) 
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous() 
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        x = selective_scan_fn(x, dt, A, B, C, self.D.float(), z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        return x

class HVMamba(nn.Module):
    def __init__(
        self,
        d_model,
        seqlen,
        d_state=16,
        d_conv=4,
        expand=2,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.seqlen = seqlen
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        # input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner*3, bias=bias, **factory_kwargs)
        
        # x_path
        self.x_path=SSM_Path(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=self.expand)
        # y_path
        self.y_path=SSM_Path(d_model=self.seqlen, d_state=d_state, d_conv=d_conv, expand=1)
        
        # z_path
        self.act = nn.SiLU()
        
        # output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


    def forward(self, hidden_states): # (B, L, D)
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xzy = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xzy = xzy + rearrange(self.in_proj.bias.to(dtype=xzy.dtype), "d -> d 1")
        x, z, y = xzy.chunk(3, dim=1)
        z = self.act(z)
        x = self.x_path(x)
        x = x * z
        y = rearrange(y, "b l d -> b d l")
        y = self.y_path(y)
        y = rearrange(y, "b d l -> b l d")
        y = y * z
        output = x + y
        output= rearrange(output, "b d l -> b l d")
        output = self.out_proj(output)
        return output

class LSEB(nn.Module):
    def __init__(self, extractor_l, extractor_s, d_model, dropout=0.1, pre_norm=True):
        super(LSEB, self).__init__()
        self.extractor_l = extractor_l
        self.extractor_s = extractor_s
        self.dropout = nn.Dropout(dropout)
        self.norm_l = RMSNorm(d_model)
        self.norm_s = RMSNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, x_l, x_s):
        # Residual connection
        res_l = x_l
        res_s = x_s
        # Pre-normalization
        if self.pre_norm:
            # print("Pre-norm in LSEB")
            x_l = self.norm_l(x_l)
            x_s = self.norm_s(x_s)
        # feature extraction
        x_l = res_l + self.dropout(self.extractor_l(x_l))
        x_s = res_s + self.dropout(self.extractor_s(x_s))
        # Post-normalization
        if not self.pre_norm:
            # print("Post-norm in LSEB")
            x_l = self.norm_l(x_l)
            x_s = self.norm_s(x_s)
        return x_l, x_s

class LSIB(nn.Module):
    def __init__(self, interactor, d_model, dropout=0.1, pre_norm=True, fuse=True):
        super(LSIB, self).__init__()

        self.interactor_l = interactor
        self.interactor_s = interactor
        self.norm_l = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        self.fuse = fuse

    def forward(self, x_l, x_s):
        # Residual connection
        res_l = x_l
        res_s = x_s
        # Pre-normalization
        if self.pre_norm:
            # print("Pre-norm in LSIB")
            x_l = self.norm_l(x_l)
            x_s = self.norm_s(x_s)
        # long-short fuse
        if self.fuse:
            new_l, attn_l = self.interactor_l(x_l, x_s, x_s, attn_mask=None, tau=None, delta=None)
            new_s, attn_s = self.interactor_s(x_s, x_l, x_l, attn_mask=None, tau=None, delta=None)
        else:
            new_l, attn_l = self.interactor_l(x_l, x_l, x_l, attn_mask=None, tau=None, delta=None)
            new_s, attn_s = self.interactor_s(x_s, x_s, x_s, attn_mask=None, tau=None, delta=None)
        x_l = res_l + self.dropout(new_l)
        x_s = res_s + self.dropout(new_s)
        # Post-normalization
        if not self.pre_norm:
            # print("Post-norm in LSIB")
            x_l = self.norm_l(x_l)
            x_s = self.norm_s(x_s)
        return x_l, x_s, attn_l, attn_s

class LEB(nn.Module):
    def __init__(self, extractor, d_model, dropout=0.1, pre_norm=True):
        super(LEB, self).__init__()
        self.extractor = extractor
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)
        self.pre_norm = pre_norm

    def forward(self, x):
        # Residual connection
        res = x
        # Pre-normalization
        if self.pre_norm:
            x = self.norm(x)
        # feature extraction
        x = res + self.dropout(self.extractor(x))
        # Post-normalization
        if not self.pre_norm:
            x = self.norm(x)
        return x    
    
class LIB(nn.Module):
    def __init__(self, interactor, d_model, dropout=0.1, pre_norm=True):
        super(LIB, self).__init__()

        self.interactor = interactor
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x):
        # Residual connection
        res = x
        # Pre-normalization
        if self.pre_norm:
            x = self.norm(x)
        # long self-attention
        new_x, attn_x = self.interactor(x, x, x, attn_mask=None, tau=None, delta=None)
        x = res + self.dropout(new_x)
        # Post-normalization
        if not self.pre_norm:
            x = self.norm(x)
        return x, attn_x

class EncoderLayer(nn.Module):
    def __init__(self, extractor_l, extractor_s, interactor, d_model, d_ff=None, dropout=0.1, activation="gelu", decomp=True, fuse=True, pre_norm=True):
        super(EncoderLayer, self).__init__()

        self.decomp = decomp
        self.fuse = fuse
        d_ff = d_ff or 4 * d_model

        if self.decomp:
            # Long-Short extractor
            self.extractor_ls = LSEB(extractor_l, extractor_s, d_model, dropout=dropout, pre_norm=pre_norm)
            # Long-Short interactor
            self.interactor_ls = LSIB(interactor, d_model, dropout=dropout, pre_norm=pre_norm, fuse=fuse)
            # Long-Short MLP
            self.mlp_long = FeedForward(d_model, d_ff, dropout=dropout, activation=activation, pre_norm=pre_norm)
            self.mlp_short = FeedForward(d_model, d_ff, dropout=dropout, activation=activation, pre_norm=pre_norm)
        else:
            # Long extractor
            self.extractor = LEB(extractor_l, d_model, dropout=dropout, pre_norm=pre_norm)
            # Long interactor
            self.interactor = LIB(interactor, d_model, dropout=dropout, pre_norm=pre_norm)
            # Long MLP
            self.mlp = FeedForward(d_model, d_ff, dropout=dropout, activation=activation, pre_norm=pre_norm)

    def forward(self, x_l, x_s):
        if self.decomp:
            x_l, x_s = self.extractor_ls(x_l, x_s)
            x_l, x_s, attn_l, attn_s = self.interactor_ls(x_l, x_s)
            x_l = self.mlp_long(x_l)
            x_s = self.mlp_short(x_s)
            return x_l, x_s, attn_l, attn_s
        else:
            x_l = self.extractor(x_l)
            x_l, attn_l = self.interactor(x_l)
            x_l = self.mlp(x_l)
    
            return x_l, attn_l

class Encoder(nn.Module):
    def __init__(self, attn_layers, decomp = True, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm_layer = norm_layer
        self.norm_l = norm_layer
        self.norm_s = norm_layer
        self.decomp = decomp

    def forward(self, x_l, x_s):
        # x [B, L, D]
        if self.decomp:
            attns_l = []
            attns_s = []
            for attn_layer in self.attn_layers:
                x_l, x_s, attn_l, attn_s = attn_layer(x_l, x_s)
                attns_l.append(attn_l)
                attns_s.append(attn_s)

            if self.norm_layer is not None:
                x_l = self.norm_l(x_l)
                x_s = self.norm_s(x_s)

            return x_l, x_s, attns_l, attns_s
        else:
            attns_l = []
            for attn_layer in self.attn_layers:
                x_l, attn_l = attn_layer(x_l, x_s=None)
                attns_l.append(attn_l)

            if self.norm_layer is not None:
                x_l = self.norm_layer(x_l)

            return x_l, attns_l

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output