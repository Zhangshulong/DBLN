import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """The original positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_positions:
        The max number of positions.

    """

    def __init__(self, d_hid: int, n_positions: int = 1000):
        super().__init__()
        pe = torch.zeros(n_positions, d_hid, requires_grad=False).float()
        position = torch.arange(0, n_positions).float().unsqueeze(1)
        div_term = (torch.arange(0, d_hid, 2).float() * -(torch.log(torch.tensor(10000)) / d_hid)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pos_table", pe)

    def forward(
        self,
        x: torch.Tensor,
        dim: int = 1,
        return_only_pos: bool = False,
    ) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        dim:
            The dimension to add the positional encoding.

        return_only_pos:
            Whether to return only the positional encoding.

        Returns
        -------
        If return_only_pos is True:
            pos_enc:
                The positional encoding.
        else:
            x_with_pos:
                Output tensor, the input tensor with the positional encoding added.
        """
        pos_enc = self.pos_table[:, : x.size(dim)].clone().detach()

        if return_only_pos:
            return pos_enc

        x_with_pos = x + pos_enc
        return x_with_pos

class SaitsEmbedding(nn.Module):
    """The embedding method from the SAITS paper :cite:`du2023SAITS`.

    Parameters
    ----------
    d_in :
        The input dimension.

    d_out :
        The output dimension.

    with_pos :
        Whether to add positional encoding.

    n_max_steps :
        The maximum number of steps.
        It only works when ``with_pos`` is True.

    dropout :
        The dropout rate.

    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        with_pos: bool,
        n_max_steps: int = 1000,
        dropout: float = 0,
    ):
        super().__init__()
        self.with_pos = with_pos
        self.dropout_rate = dropout

        self.embedding_layer = nn.Linear(d_in, d_out)
        self.position_enc = PositionalEncoding(d_out, n_positions=n_max_steps) if with_pos else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, X, missing_mask=None):
        if missing_mask is not None:
            X = torch.cat([X, missing_mask], dim=2)

        X_embedding = self.embedding_layer(X)

        if self.with_pos:
            X_embedding = self.position_enc(X_embedding)
        if self.dropout_rate > 0:
            X_embedding = self.dropout(X_embedding)

        return X_embedding


class EvidenceMachineKernel(nn.Module):
    def __init__(self, C, F):
        super().__init__()
        self.C = C
        self.F = 2**F
        self.C_weight = nn.Parameter(torch.randn(self.C, self.F))
        self.C_bias = nn.Parameter(torch.randn(self.C, self.F))

    def forward(self, x):
        x = torch.einsum("btc,cf->btcf", x, self.C_weight) + self.C_bias
        return x


class BackboneTEFN(nn.Module):
    def __init__(
        self,
        n_steps,
        n_features,
        n_pred_steps,
        n_fod,
    ):
        super().__init__()

        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_fod = n_fod

        self.T_model = EvidenceMachineKernel(self.n_steps + self.n_pred_steps, self.n_fod)
        self.C_model = EvidenceMachineKernel(self.n_features, self.n_fod)

    def forward(self, X) -> torch.Tensor:
        X = self.T_model(X.permute(0, 2, 1)).permute(0, 2, 1, 3) + self.C_model(X)
        X = torch.einsum("btcf->btc", X)
        return X


class TEFN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_steps: int,
        n_features: int,
        n_fod: int,
        dropout: float,
    ):
        super().__init__()
        self.n_fod = n_fod
        self.saits_embedding = SaitsEmbedding(
            n_features * 2,
            n_features,
            with_pos=False,
        )
        self.model = BackboneTEFN(
            n_steps,
            n_features,
            0,
            n_fod,
        )
        self.activation_func = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(n_features * n_steps, n_classes)

    def forward(self,inputs):
        # X, missing_mask = inputs["X"], inputs["missing_mask"]
        X= inputs
        missing_mask = torch.randint(0, 2, X.shape).float().to(X.device)
        bz = X.shape[0]

        enc_out = self.saits_embedding(X, missing_mask)

        # TEFN processing
        out = self.model(enc_out)
        out = self.activation_func(out)
        out = self.dropout(out)

        logits = self.output_projection(out.reshape(bz, -1))

        return logits
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        n_class = configs.num_class
        n_steps = configs.seq_len
        n_features = configs.enc_in
        n_fod = configs.e_layers
        dropout = configs.dropout
        self.model = TEFN(
            n_classes=n_class,
            n_steps=n_steps,
            n_features=n_features,
            n_fod=n_fod,
            dropout=dropout,
        )
    def forward(self, x, x_mark=None, seg_mask=None, attn_mask=None):
        output = self.model(x)
        return output

