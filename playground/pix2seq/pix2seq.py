# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer
from .loss import sigmoid_focal_loss
from util.box_ops import box_cxcywh_to_xyxy


class Pix2Seq(nn.Module):
    """ This is the Pix2Seq module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_bins=2000):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_bins: number of bins for each side of the input image
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.input_proj = nn.Sequential(
            nn.Linear(backbone.num_channels, hidden_dim),
            nn.LayerNorm(hidden_dim))
        self.backbone = backbone

    def forward(self, image_tensor, input_seq=None):
        """ 
            samples[0]:
            The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            samples[1]:
                targets
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all vocabulary.
                                Shape= [batch_size, num_sequence, num_vocal]
        """
        if isinstance(image_tensor, (list, torch.Tensor)):
            image_tensor = nested_tensor_from_tensor_list(image_tensor)
        features, pos = self.backbone(image_tensor)

        src, mask = features[-1].decompose()
        assert mask is not None
        src = src.flatten(2).permute(2, 0, 1)
        mask = torch.zeros_like(mask).bool()

        src = self.input_proj(src)
        if self.training:
            out = self.forward_train(src, input_seq, mask, pos[-1])
        else:
            out_logits, out_seq = self.forward_inference(src, mask, pos[-1])

        if not self.training:
            out = {'pred_seq_logits': out_logits, 'pred_seq': out_seq}
        elif self.transformer.decoder.return_intermediate:
            assert out.shape[0] > 1
            out = {'pred_seq_logits': out[-1], 'aux_seq_logits': out[:-1]}
        else:
            out = {'pred_seq_logits': out[-1]}
        return out

    def forward_train(self, src, input_seq, mask, pos):
        similarity = self.transformer(src, input_seq, mask, pos)
        return similarity

    def forward_inference(self, src, mask, pos):
        out_seq = self.transformer(src, -1, mask, pos)
        return out_seq


class SetCriterion(nn.Module):
    """
    This class computes the loss for Pix2Seq.
    """
    def __init__(self, num_classes, weight_dict, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.num_bins = args.num_bins
        self.num_vocal = args.dictionary.num_vocal
        self.loss_type = args.loss_type
        self.focal_alpha = args.focal_alpha
        empty_weight = torch.ones(self.num_vocal)
        empty_weight[-1] = args.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, target_seq):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_pos = (target_seq > -1).sum()
        num_pos = torch.as_tensor([num_pos], dtype=torch.float, device=target_seq.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pos)
        num_pos = torch.clamp(num_pos / get_world_size(), min=1).item()

        pred_seq_logits = outputs['pred_seq_logits']
        B, L, C = pred_seq_logits.shape

        if isinstance(pred_seq_logits, list) and not self.training:
            num_pred_seq = [len(seq) for seq in pred_seq_logits]
            pred_seq_logits = torch.cat(pred_seq_logits, dim=0).reshape(-1, self.num_vocal)
            target_seq = torch.cat(
                [t_seq[:p_seq] for t_seq, p_seq in zip(target_seq, num_pred_seq)], dim=0)
        elif self.loss_type != "ce_specific":
            pred_seq_logits = pred_seq_logits.reshape(-1, self.num_vocal)
            target_seq = target_seq.flatten()

        if self.loss_type == "ce":
            loss_seq = F.cross_entropy(pred_seq_logits, target_seq, weight=self.empty_weight, reduction='sum') / num_pos
        elif self.loss_type == "ce_specific":
            loss_cls = F.cross_entropy(pred_seq_logits[:, 4::5, 501: 592].reshape(-1, 91),
                                       (target_seq[:, 4::5] - 501).clamp(min=-100).flatten(),
                                       reduction='sum')
            loss_eos = F.cross_entropy(torch.cat([pred_seq_logits[:, 0::5, :501], 
                                                  pred_seq_logits[:, 0::5, 592:593]], dim=2).reshape(-1, 502),
                                       target_seq[:, 0::5].clamp(max=501).flatten(),
                                       reduction='sum')
            others_mask = torch.ones(L).bool()
            others_mask[0::5]=False
            others_mask[4::5]=False
            loss_others = F.cross_entropy(pred_seq_logits[:,others_mask][:, :, :501].reshape(-1, 501),
                                          target_seq[:, others_mask].flatten(),
                                          reduction='sum')
            loss_seq = (loss_cls + loss_eos + loss_others) / num_pos
        elif self.loss_type == "focal":
            valid_idx = target_seq > -100
            pred_seq_logits, target_seq = pred_seq_logits[valid_idx], target_seq[valid_idx]

            target_seq_onehot = torch.zeros(pred_seq_logits.shape,
                                                dtype=pred_seq_logits.dtype, layout=pred_seq_logits.layout, device=pred_seq_logits.device)
            target_seq_onehot.scatter_(1, target_seq.unsqueeze(-1), 1)
            loss_seq = sigmoid_focal_loss(pred_seq_logits, target_seq_onehot, alpha=self.focal_alpha, gamma=2) / num_pos
        else:
            raise KeyError

        # Compute all the requested losses
        losses = dict()
        losses["loss_seq"] = loss_seq

        # zychen added for intermediate
        if "aux_seq_logits" in outputs:
            for i, aux_pred_seq_logits in enumerate(outputs["aux_seq_logits"]):
                if isinstance(aux_pred_seq_logits, list) and not self.training:
                    num_pred_seq = [len(seq) for seq in aux_pred_seq_logits]
                    aux_pred_seq_logits = torch.cat(aux_pred_seq_logits, dim=0).reshape(-1, self.num_vocal)
                    target_seq = torch.cat(
                        [t_seq[:p_seq] for t_seq, p_seq in zip(target_seq, num_pred_seq)], dim=0)
                else:
                    aux_pred_seq_logits = aux_pred_seq_logits.reshape(-1, self.num_vocal)
                    target_seq = target_seq.flatten()

                loss_aux = F.cross_entropy(aux_pred_seq_logits, target_seq, weight=self.empty_weight, reduction='sum') / num_pos
                losses["loss_%d"%(i)] = loss_aux

        return losses


BASE_VOCAB_SHIFT = 100

def seq_to_bbox(seq, quantization_bins, seq_format='xyxy_name'):
    """Returns [0, 1] normalized yxyx bbox from token sequence."""
    # [batch, 5*num_instances]
    assert seq.dim() == 2, seq.shape.as_list()
    # [batch, num_instances, 1]
    if seq_format.startswith('name'):
        ymin = seq[:, 1::5, None]
        xmin = seq[:, 2::5, None]
        ymax = seq[:, 3::5, None]
        xmax = seq[:, 4::5, None]
    else:
        ymin = seq[:, 0::5, None]
        xmin = seq[:, 1::5, None]
        ymax = seq[:, 2::5, None]
        xmax = seq[:, 3::5, None]
    if seq_format in ['name_cxcywh', 'cxcywh_name']:
        ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
        ymin = ycnt - ysize//2
        xmin = xcnt - xsize//2
        ymax = ycnt + ysize//2
        xmax = xcnt + xsize//2
    quantized_box = torch.cat([xmin, ymin, xmax, ymax], -1)
    # TODO: dequantize
    quantized_box = quantized_box / (quantization_bins - 1)
    return quantized_box.clamp(min=0., max=1.)

def decode_object_seq_to_bbox(logits,
                              pred_seq,
                              quantization_bins,
                              coord_vocab_shift):
    """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.
    Assume yxyxc format with truncation at the end for any uneven extra tokens.
        Replace class tokens with argmax instead of sampling.
    Args:
        logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
        pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
        quantization_bins: `int` for bins.
        coord_vocab_shift: `int`, shifting coordinates by a specified integer.
    Returns:
        pred_class: `int` of shape (bsz, max_instances_per_image).
        pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
        pred_score: `float` of shape (bsz, max_instances_per_image).
    """
    bs, seqlen, vocab_size = logits.shape
    if seqlen % 5 != 0:  # truncate out the last few tokens.
        pred_seq = pred_seq[..., :-(seqlen % 5)]
        logits = logits[..., :-(seqlen % 5), :]
    pred_class_p = logits.softmax(dim=-1)[:, 4::5]  # (bsz, instances, vocab_size)
    mask_s1 = [0.] * BASE_VOCAB_SHIFT  # reserved.
    mask_s2 = [1.] * (coord_vocab_shift - BASE_VOCAB_SHIFT)  # labels.
    mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
    mask = torch.as_tensor(mask_s1 + mask_s2 + mask_s3, device=pred_class_p.device)
    pred_class = (pred_class_p * mask).argmax(-1, keepdim=True)
    pred_score = pred_class_p.gather(-1, pred_class).squeeze(-1)
    # pred_score = tf.reduce_sum(
    #     pred_class_p * tf.one_hot(pred_class, vocab_size), -1)
    pred_class = (pred_class.squeeze(-1) - BASE_VOCAB_SHIFT).clamp(min=0)
    pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
    return pred_class, pred_bbox, pred_score


def scale_points(points, scale):
    """Scales points.
    Args:
        points: Tensor with shape [num_points * 2], [batch, num_points * 2] or
        [batch, instances, num_points * 2] where points are organized in
        (y, x) format.
        scale: Tensor with shape [2] or [batch, 2].
    Returns:
        Tensor with same shape as points.
    """
    points_orig = points
    orig_shape = points.shape
    coords_len = points.shape[-1]
    if points.dim() == 1:
        points = points.reshape(coords_len // 2, 2)
    elif points.dim() == 2:
        points = points.reshape(-1, coords_len // 2, 2)
    else:
        points = points.reshape(-1, orig_shape[1], coords_len // 2, 2)
        scale = scale.unsqueeze(-2)
    points = points * scale
    points = points.reshape(orig_shape)
    # points = preserve_reserved_tokens(points, points_orig)
    return points


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_classes, args):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = args.num_bins
        self.max_input_size = args.max_input_size

    def forward(self, outputs, targets):
        return self.forward_tokens(outputs, targets)

    @torch.no_grad()
    def forward_tokens(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            targets:
            # target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
            #               For evaluation, this must be the original image size (before any data augmentation)
            #               For visualization, this should be the image size after data augment, but before padding
        """
        image_ids = torch.cat([t["image_id"] for t in targets], dim=0)
        orig_image_size = torch.stack([t["orig_size"] for t in targets], dim=0)
        unpadded_image_size = torch.stack([t["size"] for t in targets], dim=0)

        # Decode sequence output.
        # TODO: add parameters
        quantization_bins, coord_vocab_shift = 1000, 1000
        pred_classes, pred_bboxes, scores = decode_object_seq_to_bbox(
            outputs["pred_seq_logits"], outputs["pred_seq"], quantization_bins, coord_vocab_shift)

        # Compute coordinate scaling from [0., 1.] to actual pixels in orig image.
        image_size = torch.as_tensor([self.max_input_size, self.max_input_size], device=orig_image_size.device)
        if self.training:
            # scale points to whole image size during train.
            scale = utils.tf_float32(image_size)
        else:
            # scale points to original image size during eval.
            scale = image_size / unpadded_image_size
            scale = scale * orig_image_size
            scale = scale[:, None]
        pred_bboxes_rescaled = scale_points(pred_bboxes, scale)

        results = []
        for icls, ibox, iscore in zip(pred_classes, pred_bboxes_rescaled, scores):
            result = dict()
            result['scores'] = iscore
            result['labels'] = icls
            result['boxes'] = ibox
            results.append(result)

        return results

    @torch.no_grad()
    def forward_logits(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            targets:
            # target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
            #               For evaluation, this must be the original image size (before any data augmentation)
            #               For visualization, this should be the image size after data augment, but before padding
        """
        origin_img_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        input_img_sizes = torch.stack([t["size"] for t in targets], dim=0)
        out_seq_logits = outputs['pred_seq_logits']

        assert len(out_seq_logits) == len(origin_img_sizes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        ori_img_h, ori_img_w = origin_img_sizes.unbind(1)
        inp_img_h, inp_img_w = input_img_sizes.unbind(1)
        scale_fct = torch.stack(
            [ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h], dim=1).unsqueeze(1)

        results = []
        out_seq_logits[:, :, self.num_bins + 1 + self.num_classes] += self.eos_bias
        for b_i, pred_seq_logits in enumerate(out_seq_logits):
            seq_len = pred_seq_logits.shape[0]
            pred_seq_logits = pred_seq_logits.softmax(dim=-1)
            num_objects = seq_len // 5
            pred_seq_logits = pred_seq_logits[:int(num_objects * 5)].reshape(num_objects, 5, -1)
            pred_boxes_logits = pred_seq_logits[:, :4, :self.num_bins + 1]
            pred_class_logits = pred_seq_logits[:, 4, self.num_bins + 1: self.num_bins + 1 + self.num_classes]
            pred_eos_logits = pred_seq_logits[:, 0, self.num_bins + 1 + self.num_classes]
            is_eos = pred_boxes_logits[:, 0, :].max(dim=1)[0] < pred_eos_logits
            if is_eos.any():
                count_items = torch.nonzero(is_eos).min()
                if count_items == 0:
                    results.append(dict())
                    continue
                pred_class_logits, pred_boxes_logits = pred_class_logits[:count_items], pred_boxes_logits[:count_items]
            scores_per_image, labels_per_image = torch.max(pred_class_logits, dim=1)
            boxes_per_image = pred_boxes_logits.argmax(dim=2) * self.max_input_size / self.num_bins
            boxes_per_image = boxes_per_image * scale_fct[b_i]
            result = dict()
            result['scores'] = scores_per_image
            result['labels'] = labels_per_image
            result['boxes'] = boxes_per_image
            results.append(result)

        return results


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Pix2Seq(
        backbone,
        transformer,
        num_classes=num_classes,
        num_bins=args.num_bins)

    weight_dict = {'loss_seq': 1}
    criterion = SetCriterion(
        num_classes,
        weight_dict,
        args=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_classes, args)}

    return model, criterion, postprocessors
