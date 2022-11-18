import torch
from torch import nn
from typing import Optional
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
)
from speechbrain.nnet.activations import Swish


class TransformerAM(TransformerInterface):

    def __init__(
        self,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            **kwargs
        )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )

        # reset parameters using xavier_normal_
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def forward(self, x, lengths=None):
        """
        Arguments
        ----------
        x : tensor
            Inputs
        lengths : tensor
            Relative lengths of inputs, used to discard padding
        """

        # reshape the input vector to [Batch, Time, Fea] is a 4d vector is given
        if x.ndim == 4:
            bz, t, ch1, ch2 = x.shape
            x = x.reshape(bz, t, ch1 * ch2)

        x = self.custom_src_module(x)
        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(x)
        elif self.positional_encoding_type == "fixed_abs_sine":
            x = x + self.positional_encoding(x)  # add the encodings here
            pos_embs_encoder = None
        else:
            # No encoding
            pos_embs_encoder = None
        if lengths is not None:
            abs_len = torch.round(lengths * x.shape[1])
            src_key_padding_mask = (
                torch.arange(x.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )
        else:
            src_key_padding_mask = None
        encoder_out, _ = self.encoder(
            src=x,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        return encoder_out


if __name__ == "__main__":
    trafo = TransformerAM(768)
    inp = torch.randn(10,60,768)
    print(trafo(inp).shape)
