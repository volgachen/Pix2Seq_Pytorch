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
                 activation="relu", normalize_before=False, args=None):
                 # num_vocal=2094, max_objects=100, pred_eos=False, return_intermediate_dec=False, query_pos=False, drop_cls=0.):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, drop_path, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, drop_path, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=args.return_intermediate_dec)
        self._reset_parameters()

        self.num_vocal = args.dictionary.num_vocal
        if args.classifier_norm:
            self.classifier_norm = nn.LayerNorm(d_model)
        else:
            self.classifier_norm = None
        self.vocal_classifier = nn.Linear(d_model, self.num_vocal)
        self.det_embed = nn.Embedding(1, d_model)
        self.vocal_embed = nn.Embedding(self.num_vocal - 2, d_model)
        self.num_bins = args.num_bins

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers

        self.max_objects = args.max_objects
        # zychen: for query_pos
        if args.query_pos:
            self.query_pos = nn.Embedding(self.max_objects * 5 + 1, d_model)
        else:
            self.query_pos = None
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
        bs = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, bs, 1) if self.query_pos is not None else None

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        pre_kv = [torch.as_tensor([[], []], device=memory.device)
                  for _ in range(self.num_decoder_layers)]

        if self.training:
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
            if self.classifier_norm is not None:
                hs = self.classifier_norm(hs)
            pred_seq_logits = self.vocal_classifier(hs.transpose(1, 2))
            return pred_seq_logits
        else:
            end = torch.zeros(bs).bool().to(memory.device)
            end_lens = torch.zeros(bs).long().to(memory.device)
            input_embed = self.det_embed.weight.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1)
            result_probs, result_tokens = [], []
            for seq_i in range(self.max_objects * 5):
                query_pos = self.query_pos.weight[seq_i].view(1,1,-1).repeat(1,bs,1) if self.query_pos is not None else None
                hs, pre_kv = self.decoder(
                    input_embed,
                    memory,
                    memory_key_padding_mask=mask,
                    pos=pos_embed,
                    pre_kv_list=pre_kv,
                    query_pos=query_pos)
                if self.classifier_norm is not None:
                    hs = self.classifier_norm(hs)
                similarity = self.vocal_classifier(hs[-1])

                valid = torch.ones((self.num_vocal - 1,), device=memory.device, dtype=torch.bool)
                if seq_i % 5 == 4:
                    valid[:501] = False
                else:
                    valid[501:592] = False
                if seq_i % 5 > 0:
                    valid[592] = False
                similarity[:, :, self.num_vocal-2] += self.eos_bias
                pred_token, pred_prob = NucleusSearch(similarity[-1, :, :self.num_vocal - 1], p=self.eval_p, valid=valid.unsqueeze(0).repeat(bs, 1))

                result_tokens.append(pred_token)
                result_probs.append(pred_prob)
                input_embed = self.vocal_embed(pred_token.clamp(max=self.num_bins+91).view(1, -1))

            return torch.stack(result_probs, dim=1), torch.stack(result_tokens, dim=1)


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

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
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

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
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


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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


def NucleusSearch(logits, p=None, m=None, temp=None, valid=None):
    """
    logits: bs * V
    """
    if valid is not None:
        logits[~valid] = float("-inf")
    if temp is not None:
        samp_probs = F.softmax(logits.div_(temp), dim=-1)
    else:
        samp_probs = F.softmax(logits, dim=-1)
    if valid is not None:
        samp_probs[~valid] = 0.

    if p is not None:
        sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        sorted_samp_probs = sorted_probs.clone()
        sorted_samp_probs[sorted_indices_to_remove] = 0
        if m is not None:
            sorted_samp_probs.div_(sorted_samp_probs.sum(1).unsqueeze(1))
            sorted_samp_probs.mul_(1-m)
            sorted_samp_probs.add_(sorted_probs.mul(m))
        sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
        next_tokens = sorted_indices.gather(1, sorted_next_indices)
        next_probs  = sorted_probs.gather(1, sorted_next_indices)
    return next_tokens, next_probs
