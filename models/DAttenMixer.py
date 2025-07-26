__all__ = ['DAttenMixer']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.DEncoder_backbone  import PatchTST_backbone
from layers.PatchMixer_backbone import Backbone as PatchMixer_backbone
from layers.decomp import DECOMP
from layers.network_mlp import NetworkMLP
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        self.nvals = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        seq_len = configs.seq_len
        pred_len = configs.pred_len

        # Moving Average
        ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(patch_num * d_model, int(target_window * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(target_window * 2), target_window),
            nn.Dropout(self.head_dropout)
        )
        self.W_P = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(self.dropout)

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = DECOMP(ma_type, alpha, beta)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                               patch_len=patch_len, stride=stride,
                                               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                               attn_dropout=attn_dropout,
                                               dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                               padding_var=padding_var,
                                               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                               store_attn=store_attn,
                                               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                               head_dropout=head_dropout, padding_patch=padding_patch,
                                               pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                               revin=revin, affine=affine,
                                               subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                               patch_len=patch_len, stride=stride,
                                               max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                               n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                               attn_dropout=attn_dropout,
                                               dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                               padding_var=padding_var,
                                               attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                               store_attn=store_attn,
                                               pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                               head_dropout=head_dropout, padding_patch=padding_patch,
                                               pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                               revin=revin, affine=affine,
                                               subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model = PatchMixer_backbone(configs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           patch_len=patch_len, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        #print(f"[Debug] Shape of x before reshape: {x.shape}, Elements: {x.numel()}")
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2,
                                                                                 1)  # x: [Batch, Channel, Input length]
            # norm
            if self.revin:
                res_init = res_init.permute(0, 2, 1)
                trend_init = trend_init.permute(0, 2, 1)
                res_init = self.revin_layer(res_init, 'norm')
                trend_init = self.revin_layer(trend_init, 'norm')
                res_init = res_init.permute(0, 2, 1)
                trend_init = trend_init.permute(0, 2, 1)
            #print(f"[Debug] Shape of trend_init before reshape: {trend_init.shape}, Elements: {trend_init.numel()}")
            res= self.model_res(res_init)
            #print(f"[Debug] Shape of res before reshape: {res.shape}, Elements: {res.numel()}")
            trend = self.model_trend(trend_init) # [Batch * Channel, patch_num, d_model]
            #print(f"[Debug] Shape of trend before reshape: {trend.shape}, Elements: {trend.numel()}")
            x = torch.cat((res, trend), dim=2)  # [Batch * Channel, patch_num, d_model * 2]

            x = self.W_P(x)  # [Batch * Channel, patch_num, d_model]
            x = self.dropout(x)
            x= self.head1(x)  # [Batch * Channel, context_window]
            x = x.reshape(x.shape[0]//self.nvals, self.nvals, -1)  # [Batch, Channel, context_window]
            x = x.permute(0, 2, 1)  # x: [Batch, context_window, Channel]
            if self.revin:
                x = self.revin_layer(x, 'denorm')
            #x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x