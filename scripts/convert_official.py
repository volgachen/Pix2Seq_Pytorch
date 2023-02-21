import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp
import torch

var_list = cp.list_variables('coco_det_finetune_resnet_640x640_ckpt-74844')
new_dict = {}

SKIP_NAMES = ["_CHECKPOINTABLE_OBJECT_GRAPH", "global_step/.ATTRIBUTES/VARIABLE_VALUE", "save_counter/.ATTRIBUTES/VARIABLE_VALUE"]
oneMapping = {
    "model/decoder/ar_decoder.Sseq_pos_embedding/.ATTRIBUTES/VARIABLE_VALUE": "transformer.query_pos_embed",
    "model/decoder/ar_decoder.Stoken_embedding/.ATTRIBUTES/VARIABLE_VALUE": "transformer.vocab_embed.weight",
    "model/decoder/ar_decoder.Soutp_bias/.ATTRIBUTES/VARIABLE_VALUE": "transformer.vocab_bias",
}

for (old_name, shape) in var_list:
    if "optimizer/" in old_name or old_name in SKIP_NAMES:
        continue
    new_name = old_name
    # common
    new_name = new_name.replace("/kernel/", "/weight/")
    new_name = new_name.replace("/gamma/", "/weight/")
    new_name = new_name.replace("/beta/", "/bias/")
    new_name = new_name.replace("/moving_mean/", "/running_mean/")
    new_name = new_name.replace("/moving_variance/", "/running_var/")
    new_name = new_name.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")

    if old_name.startswith("model/encoder/resnet/"):
        new_name = new_name.replace("model/encoder/resnet/", "backbone.0.body.")
        # stem
        new_name = new_name.replace("initial_conv_relu_max_pool/0/conv2d/", "conv1.")
        new_name = new_name.replace("initial_conv_relu_max_pool/2/bn/", "bn1.")

        # blocks
        new_name = new_name.replace("block_groups/0/", "layer1.")
        new_name = new_name.replace("block_groups/1/", "layer2.")
        new_name = new_name.replace("block_groups/2/", "layer3.")
        new_name = new_name.replace("block_groups/3/", "layer4.")
        new_name = new_name.replace(".layers/", ".")

        new_name = new_name.replace("/projection_layers/0/conv2d/", ".downsample.0.")
        new_name = new_name.replace("/projection_layers/1/bn/", ".downsample.1.")
        for idc in range(3):
            id_old = idc * 3
            id_new = idc + 1
            new_name = new_name.replace(f"/conv_relu_dropblock_layers/{id_old}/conv2d/", f".conv{id_new}.")
            new_name = new_name.replace(f"/conv_relu_dropblock_layers/{id_old+1}/bn/", f".bn{id_new}.")
    elif old_name.startswith("model/encoder/stem_"):
        new_name = new_name.replace("model/encoder/stem_projection/", "input_proj.0.")
        new_name = new_name.replace("model/encoder/stem_ln/", "input_proj.1.")
    elif old_name.startswith("model/encoder/transformer_encoder/enc_layers/"):
        new_name = new_name.replace("model/encoder/transformer_encoder/enc_layers/", "transformer.encoder.layers.")
        new_name = new_name.replace("/mha_ln/", ".norm1.")
        # TODO: qkv here
        new_name = new_name.replace("/mha/", ".self_attn.")
        new_name = new_name.replace("_output_dense/", "out_proj.")
        new_name = new_name.replace("/mlp/layernorms/0/", ".norm2.")
        new_name = new_name.replace("/mlp/mlp_layers/0/dense1/", ".linear1.")
        new_name = new_name.replace("/mlp/mlp_layers/0/dense2/", ".linear2.")
    elif old_name.startswith("model/decoder/decoder/dec_layers/"):
        new_name = new_name.replace("model/decoder/decoder/dec_layers/", "transformer.decoder.layers.")
        new_name = new_name.replace("/self_ln/", ".norm1.")
        # TODO: qkv here
        new_name = new_name.replace("/self_mha/", ".self_attn.")
        new_name = new_name.replace("_output_dense/", "out_proj.")
        new_name = new_name.replace("/cross_ln/", ".norm2.")
        # TODO: qkv here
        new_name = new_name.replace("/cross_mha/", ".multihead_attn.")
        new_name = new_name.replace("/mlp/layernorms/0/", ".norm3.")
        new_name = new_name.replace("/mlp/mlp_layers/0/dense1/", ".linear1.")
        new_name = new_name.replace("/mlp/mlp_layers/0/dense2/", ".linear2.")
    elif old_name.startswith("model/encoder/output_ln/"):
        new_name = new_name.replace("model/encoder/output_ln/", "transformer.encoder.norm.")
    elif old_name.startswith("model/decoder/output_ln/"):
        new_name = new_name.replace("model/decoder/output_ln/", "transformer.decoder.norm.")
    elif old_name.startswith("model/proj"):
        new_name = new_name.replace("model/proj/", "transformer.encoder_proj.")
        new_name = new_name.replace("model/proj_ln/", "transformer.encoder_proj_ln.")
        new_name = new_name.replace("model/proj_mlp/layernorms/0/", "transformer.encoder_mlp_ln.")
        new_name = new_name.replace("model/proj_mlp/mlp_layers/0/dense1/", "transformer.encoder_mlp_linear1.")
        new_name = new_name.replace("model/proj_mlp/mlp_layers/0/dense2/", "transformer.encoder_mlp_linear2.")
    elif old_name in oneMapping:
        new_name = oneMapping[old_name]
    else:
        print(f"Warning: {old_name} is not processed")
    if "_dense/" in new_name:
        # print(f"Skip {new_name} currently")
        continue

    old_value = torch.as_tensor(cp.load_variable('coco_det_finetune_resnet_640x640_ckpt-74844', old_name))
    # value process
    if old_value.dim() == 4: # conv
        new_value = old_value.permute(3, 2, 0, 1)
    elif old_value.dim() == 3: # out_proj
        new_value = old_value.permute(2, 0, 1).flatten(1, 2)
    elif old_value.dim() == 2: # linear
        if new_name.endswith(".weight") and "embed" not in new_name:
            new_value = old_value.transpose(0, 1)
        else:
            print(f"2D param not convert: {new_name}")
            new_value = old_value
    else:
        new_value = old_value
    #if new_name == "input_proj.0.weight":
    #    new_value = new_value.unsqueeze(-1).unsqueeze(-1)
    new_dict[new_name] = new_value
    print(new_name, new_value.shape)

old_prefix = ["model/encoder/transformer_encoder/enc_layers/{}/mha/{}kernel/.ATTRIBUTES/VARIABLE_VALUE",
               "model/encoder/transformer_encoder/enc_layers/{}/mha/{}bias/.ATTRIBUTES/VARIABLE_VALUE",
               "model/decoder/decoder/dec_layers/{}/self_mha/{}kernel/.ATTRIBUTES/VARIABLE_VALUE",
               "model/decoder/decoder/dec_layers/{}/self_mha/{}bias/.ATTRIBUTES/VARIABLE_VALUE",
               "model/decoder/decoder/dec_layers/{}/cross_mha/{}kernel/.ATTRIBUTES/VARIABLE_VALUE",
               "model/decoder/decoder/dec_layers/{}/cross_mha/{}bias/.ATTRIBUTES/VARIABLE_VALUE"]
new_prefix = ["transformer.encoder.layers.{}.self_attn.in_proj_weight",
               "transformer.encoder.layers.{}.self_attn.in_proj_bias",
               "transformer.decoder.layers.{}.self_attn.qkv.weight",
               "transformer.decoder.layers.{}.self_attn.qkv.bias",
               "transformer.decoder.layers.{}.multihead_attn.in_proj_weight",
               "transformer.decoder.layers.{}.multihead_attn.in_proj_bias"]

# q/k/v_dense in keras: (in_dim, num_heads, dim_per_head)
# qkv in torch: (out_dim, in_dim)

for old_p, new_p in zip(old_prefix, new_prefix):
    for i in range(6):
        qkv = []
        for key in ["_query_dense/", "_key_dense/", "_value_dense/"]:
            old_name = old_p.format(i, key)
            if "weight" in new_p:
                qkv.append(torch.as_tensor(cp.load_variable('coco_det_finetune_resnet_640x640_ckpt-74844', old_name)).permute(1, 2, 0).flatten(0, 1))
            else:
                qkv.append(torch.as_tensor(cp.load_variable('coco_det_finetune_resnet_640x640_ckpt-74844', old_name)).flatten())
        new_name = new_p.format(i)
        new_v = torch.cat(qkv, dim=0)
        new_dict[new_name] = new_v
        print(new_name, new_v.shape)

torch.save(new_dict, "debug.pth")
