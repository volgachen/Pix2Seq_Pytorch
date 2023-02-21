# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention_layer import Attention


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, drop_path=0.1,
                 activation="gelu", normalize_before=False, args=None):
                 # num_vocal=2094, max_objects=100, pred_eos=False, return_intermediate_dec=False, query_pos=False, drop_cls=0.):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, drop_path, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model, eps=1e-6) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_proj = nn.Linear(d_model, d_model)
        self.encoder_proj_ln = nn.LayerNorm(d_model, eps=1e-6)

        self.encoder_mlp_ln = nn.LayerNorm(d_model, eps=1e-6)
        self.encoder_mlp_linear1 = nn.Linear(d_model, dim_feedforward)
        self.encoder_mlp_linear2 = nn.Linear(dim_feedforward, d_model)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, drop_path, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=args.return_intermediate_dec)
        self._reset_parameters()

        # TODO: add parameters
        self.vocab_embed = nn.Embedding(3000, d_model)
        self.vocab_bias = nn.Parameter(torch.zeros(3000))
        self.query_pos_embed = nn.Parameter(torch.zeros(512, d_model))

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers

        # zychen: for sampling
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.temperature = args.temperature

        # zychen: for drop_cls
        self.drop_cls = args.drop_cls
        self.eval_p = args.eval_p
        self.eos_bias = args.eos_bias
        print(f"Build with return_intermediate_dec: {args.return_intermediate_dec}, query_pos: {args.query_pos}, drop_cls: {args.drop_cls}")
        print("Eval with", self.eval_p)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, input_seq, mask, pos_embed):
        """
        Args:
            src: shape[B, C, H, W]
            input_seq: shape[B, 501, C] for training and shape[B, 1, C] for inference
            mask: shape[B, H, W]
            pos_embed: shape[B, C, H, W]
        """
        # flatten NxCxHxW to HWxNxC
        bs = src.shape[1]
        mask = mask.flatten(1)

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        src = src + pos_embed
        memory = self.encoder(src, src_key_padding_mask=None, pos=None)
        memory = self.encoder_proj_ln(self.encoder_proj(memory))
        # linear1/2
        memory = memory + pos_embed
        memory = memory + self.encoder_mlp_linear2(F.gelu(self.encoder_mlp_linear1(self.encoder_mlp_ln(memory))))
        query_pos = self.query_pos_embed.unsqueeze(1).repeat(1, bs, 1)
        pre_kv = [torch.as_tensor([[], []], device=memory.device)
                  for _ in range(self.num_decoder_layers)]

        if self.training:
            # TODO: not implemented
            input_embed = torch.cat(
                [self.det_embed.weight.unsqueeze(0).repeat(bs, 1, 1),
                 self.vocal_embed(input_seq)], dim=1)
            input_embed = input_embed.transpose(0, 1)
            if self.drop_cls > 0.:
                cls_mask = torch.rand((self.max_objects, bs, 1), device=input_embed.device) > self.drop_cls
                input_embed[5::5, :, :] *= cls_mask
            num_seq = input_embed.shape[0]
            self_attn_mask = torch.triu(torch.ones((num_seq, num_seq)), diagonal=1).bool(). \
                to(input_embed.device)
            hs, pre_kv = self.decoder(
                input_embed,
                memory,
                memory_key_padding_mask=mask,
                pos=pos_embed,
                pre_kv_list=pre_kv,
                self_attn_mask=self_attn_mask,
                query_pos=query_pos)
            pred_seq_logits = self.vocal_classifier(hs.transpose(1, 2))
            return pred_seq_logits
        else:
            end = torch.zeros(bs).bool().to(memory.device)
            end_lens = torch.zeros(bs).long().to(memory.device)
            input_embed = self.vocab_embed(torch.as_tensor([[10]]).to(memory.device)).repeat(1,bs,1)
            result_logits, result_tokens = [], []
            for step in range(500):
                query_pos = self.query_pos_embed[step].view(1,1,-1).repeat(1,bs,1)
                hs, pre_kv = self.decoder(
                    input_embed + query_pos,
                    memory,
                    memory_key_padding_mask=None,
                    pos=None,
                    pre_kv_list=pre_kv,
                    query_pos=None) # 1, 1, bs, c
                next_logits = hs[-1, -1] @ self.vocab_embed.weight.transpose(0, 1) + self.vocab_bias # 1, bs, 3000

                # default sample
                sampling_logits = next_logits / self.temperature
                sampling_logits = top_logits(sampling_logits, k=self.top_k, p=self.top_p)
                # next_token = torch.distributions.categorical.Categorical(logits=sampling_logits).sample()
                next_token = torch.multinomial(sampling_logits.softmax(-1), 1)[:, 0]

                result_tokens.append(next_token)
                result_logits.append(next_logits)
                input_embed = self.vocab_embed(next_token.unsqueeze(0))
            return torch.stack(result_logits, dim=1), torch.stack(result_tokens, dim=1)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, memory_key_padding_mask, pos, pre_kv_list=None, self_attn_mask=None, query_pos=None):
        output = tgt
        cur_kv_list = []
        intermediate = []
        for layer, pre_kv in zip(self.layers, pre_kv_list):
            output, cur_kv = layer(
                output,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                self_attn_mask=self_attn_mask,
                pre_kv=pre_kv,
                query_pos=query_pos)
            cur_kv_list.append(cur_kv)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.return_intermediate:
            return torch.stack(intermediate), cur_kv_list

        # Original, no intermediate
        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0), cur_kv_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, drop_path=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = DropPath(drop_path, batch_dim=1) if drop_path > 0. else nn.Identity()
        self.dropout2 = DropPath(drop_path, batch_dim=1) if drop_path > 0. else nn.Identity()

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_key_padding_mask, pos)
        return self.forward_post(src, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, drop_path=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = DropPath(dropout, batch_dim=1) if drop_path > 0. else nn.Identity()
        self.dropout2 = DropPath(dropout, batch_dim=1) if drop_path > 0. else nn.Identity()
        self.dropout3 = DropPath(dropout, batch_dim=1) if drop_path > 0. else nn.Identity()

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            tgt,
            memory,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
            query_pos=None,
    ):
        tgt2, pre_kv = self.self_attn(self.with_pos_embed(tgt, query_pos), pre_kv=pre_kv, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, pre_kv

    def forward_pre(
            self,
            tgt,
            memory,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
            query_pos=None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, pre_kv = self.self_attn(self.with_pos_embed(tgt2, query_pos), pre_kv=pre_kv, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, pre_kv

    def forward(
            self,
            tgt,
            memory,
            memory_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
            query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_key_padding_mask, pos, self_attn_mask, pre_kv, query_pos=query_pos)
        return self.forward_post(tgt, memory, memory_key_padding_mask, pos, self_attn_mask, pre_kv, query_pos=query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        drop_path = args.drop_path,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        args=args,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def drop_path(x, drop_prob: float = 0., training: bool = False, batch_dim: int = 0):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = [1] * x.ndim  # work with diff dim tensors, not just 2D ConvNets
    shape[batch_dim] = x.shape[batch_dim]
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, batch_dim=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.batch_dim)


def top_logits(logits: torch.Tensor,
               k: int = 0,
               p: float = 1.0,
               mask: float = -1e10) -> torch.Tensor:
    """Remove low probability logits via masking.
    Args:
        logits: class logits in shape of (batch size, total_classes).
        k: specifying top k largest logits to keep.
        p: specifying a probability for finding a minimum set of largest
        logits to keep, where their cumulative probability is no less than p
        (actually in the following version, it is "...cumulative probability is
        the largest but no more than p").
        mask: an value that's used to replace logits that don't satisfy the
        keep conditions.
    Returns:
        logits where low probability ones are replaced with mask.
    """
    mask = torch.ones_like(logits) * mask
    if k > 0:
        min_logits = logits.topk(k=k, dim=-1)[0][..., -1:]
        logits = torch.where(logits < min_logits, mask, logits)
    if p < 1.:
        sorted_logits = logits.sort(-1, descending=True).values
        cum_probs = torch.cumsum(sorted_logits.softmax(-1), -1)
        min_logits = - torch.max(
            torch.where(cum_probs <= p, -sorted_logits, mask), -1, keepdim=True).values
        min_logits = torch.minimum(min_logits, sorted_logits[:, :1])
        logits = torch.where(logits < min_logits, mask, logits)
    return logits
