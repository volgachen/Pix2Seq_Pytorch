import torch

def get_angles(pos, i, dim):
    angle_rates = 1 / (10000. ** (2 * (i//2) / dim))
    return pos * angle_rates


def positional_encoding(coords, dim):
    """coords in (bsz, size), return (bsz, size, dim)."""
    angle_rads = get_angles(coords[..., None],
                            torch.arange(dim)[None, None, :],
                            dim)

    # apply sin to even indices in the array; 2i
    angle_rads1 = angle_rads[:, :, 0::2].sin()

    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = angle_rads[:, :, 1::2].cos()

    pos_encoding = torch.cat([angle_rads1, angle_rads2], -1)

    return pos_encoding


def get_1d_position_codes(seqlen, out_dim, normalization_max=6.2831852):
    """Get 2d positional embedding with sin/cos codes.
    Args:
      seqlen: a `int` specifying the length of the sequence.
      out_dim: a `int` specifying the output dimension of the encoding.
      normalization_max: normalize coordinates between [0, normalization_max].
        If None, raw coordinates from 0 to seqlen will be used.
    Returns:
      positional code of shape (1, seqlen, out_dim)
    """
    coords = torch.arange(seqlen)
    if normalization_max is not None:
      coords = coords / (seqlen - 1) * normalization_max
    coords = positional_encoding(coords, out_dim)
    return coords


def get_2d_position_codes(height, width, out_dim, normalization_max=6.2831852):
    """Get 2d positional embedding with sin/cos codes.
    Args:
      height: a `int` specifying the height of the 2d image / feature map.
      width: a `int` specifying the width of the 2d image / feature map.
      out_dim: a `int` specifying the output dimension of the encoding.
        Must be divisible by 2.
      normalization_max: normalize coordinates between [0, normalization_max].
        If None, raw coordinates from 0 to height/width will be used.
    Returns:
      positional code of shape (1, height, width, out_dim)
    """
    y_coords = torch.arange(height).to(torch.float)
    if normalization_max is not None:
        y_coords = (
            y_coords / (height - 1) * normalization_max)
    y_coords = positional_encoding(y_coords, out_dim//2)
    y_coords = y_coords[:, :, None]
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)

    x_coords = torch.arange(width).to(torch.float)
    if normalization_max is not None:
        x_coords = (
            x_coords / (width - 1) * normalization_max)
    x_coords = positional_encoding(x_coords, out_dim//2)
    x_coords = x_coords[:, None]
    x_coords = torch.cat([torch.zeros_like(x_coords), x_coords], -1)

    return y_coords + x_coords

