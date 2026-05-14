if __name__ == '__main__':
    import sys
    sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat

from models.tsm_resnet import TSMResNetV2

# from config.const import TRAIN_SIZE
TRAIN_SIZE=(24, 256, 256, 3)


def convert_grid_coordinates(
    coords,
    input_grid_size,
    output_grid_size,
):
  if isinstance(input_grid_size, tuple):
    input_grid_size = torch.tensor(input_grid_size, device=coords.device)
  if isinstance(output_grid_size, tuple):
    output_grid_size = torch.tensor(output_grid_size, device=coords.device)

  position_in_grid = coords
  position_in_grid = position_in_grid * output_grid_size / input_grid_size

  return position_in_grid


def soft_argmax_heatmap(
    softmax_val,
    threshold = 5,
):
    """
        softmax_val: Tensor shape of ..., H, W
        threshold: Threshold for the soft argmax operation.
    """
    orig_shape = softmax_val.shape[:-2]

    softmax_val = rearrange(softmax_val, '... h w -> (...) h w')
    B, H, W = softmax_val.shape
    x, y = torch.meshgrid(
        torch.arange(W),
        torch.arange(H),
        indexing='xy'
    )
    coords = torch.stack((x, y), dim=-1).to(softmax_val.device) # H, W, 2
    argmax_pos = torch.argmax(softmax_val.flatten(-2, -1), dim=-1) # (...)
    coords_flatten = rearrange(coords, 'h w c -> (h w) c')
    pos = coords_flatten[argmax_pos] # (..., 2)

    pos = rearrange(pos, 'b c -> b () () c')
    coords = rearrange(coords, 'h w c -> () h w c')

    valid = (coords - pos).pow(2).sum(-1, keepdim=True) < threshold ** 2
    weighted_sum = (coords * valid * softmax_val[..., None]).sum(dim=(-2, -3), keepdim=True)

    sum_of_weights = (valid.squeeze(-1) * softmax_val).sum(dim=(-1, -2))[..., None, None, None]
    sum_of_weights = torch.maximum(sum_of_weights, torch.full_like(sum_of_weights, 1e-12))

    return (weighted_sum / sum_of_weights).reshape(orig_shape + (2,))


def heatmaps_to_points(
    all_pairs_softmax,
    image_shape,
    threshold = 5,
    query_points = None,
):
    """Given a batch of heatmaps, compute a soft argmax.

    If query points are given, constrain that the query points are returned
    verbatim.

    Args:
        all_pairs_softmax: A set of heatmaps, of shape [batch, num_points, time,
        height, width].
        image_shape: The shape of the original image that the feature grid was
        extracted from.  This is needed to properly normalize coordinates. [T, H, W]
        threshold: Threshold for the soft argmax operation.
        query_points (optional): If specified, we assume these points are given as
        ground truth and we reproduce them exactly.  This is a set of points of
        shape [batch, num_points, 3], where each entry is [x, y, t] in frame/
        raster coordinates.

    Returns:
        predicted points, of shape [batch, num_points, time, 2], where each point is
        [x, y] in raster coordinates.  These are the result of a soft argmax ecept
        where the query point is specified, in which case the query points are
        returned verbatim.
    """
    # soft_argmax_heatmap operates over a single heatmap.  We vmap it across
    # batch, num_points, and frames.
    out_points = soft_argmax_heatmap(all_pairs_softmax, threshold)
    # out_points is now [batch, num_points, time, 2]

    feature_grid_shape = all_pairs_softmax.shape[1:]
    # Note: out_points is now [x, y]; we need to divide by [width, height].
    # image_shape[2] is width and image_shape[1] is height.
    out_points = convert_grid_coordinates(
        out_points,
        feature_grid_shape[3:1:-1],
        image_shape[2:0:-1],
    )

    if query_points is not None:
        # The [..., 0:1] is because we only care about the frame index.
        query_frame = convert_grid_coordinates(
            query_points,
            image_shape[::-1],
            feature_grid_shape[-1:0:-1],
        )[..., 2:3] # [batch, num_points, 1]
        is_query_point = (
            torch.round(query_frame).long() == torch.arange(
                image_shape[0], device=query_frame.device
            ).view(1, 1, -1)
        )
        out_points = out_points * (
            1.0 - is_query_point[:, :, :, None].float()
        ) + query_points[:, :, None, :2] * is_query_point[:, :, :, None].float()
    return out_points


def interp(x, y):
    """
        Bilinear interpolation.
        x: Grid of features to be interpolated, of shape [..., time, height, width] or [..., height, width]
        y: Points to be interpolated, of shape [..., num_points, 3] or [..., num_points, 2]
        returns:
            Interpolated features, of shape [..., num_points]
    """
    if y.shape[-1] == 3:
        *orig, T, H, W = x.shape
        # y = torch.cat([y[..., :2], y[..., 2:]], dim=-1)
        y = y / torch.tensor([W - 1, H - 1, T - 1], device=y.device) * 2 - 1
        x = rearrange(x, '... t h w -> (...) () t h w')
        y = rearrange(y, '... n c -> (...) () () n c')

        result = F.grid_sample(x, y, mode='bilinear', align_corners=True, padding_mode='border')
        result = rearrange(result, '... () () () n -> ... n')

    else:
        *orig, H, W = x.shape
        y = y / torch.tensor([W - 1, H - 1], device=y.device) * 2 - 1
        x = rearrange(x, '... h w -> (...) () h w')
        y = rearrange(y, '... n c -> (...) () n c')

        result = F.grid_sample(x, y, mode='bilinear', align_corners=True, padding_mode='border')
        result = rearrange(result, '... () () n -> ... n')
    
    return result.reshape(*orig, y.shape[-2])


class TAPNet(nn.Module):
    def __init__(self,
        feature_grid_stride = 8,
        softmax_temperature = 10., 
        normalize_fn = 'batchnorm'
    ):
        super().__init__()

        self.softmax_temperature = softmax_temperature

        self.tsm_resnet = TSMResNetV2(
            channel_shift_fraction=[0.125, 0.125, 0., 0.],
            output_stride=feature_grid_stride,
            normalize_fn=normalize_fn,
        )

        self.hid1 = nn.Conv3d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = (1, 3, 3),
            stride = (1, 1, 1),
            padding = (0, 1, 1),
        )
        self.hid2 = nn.Conv3d(
            in_channels = 16,
            out_channels = 1,
            kernel_size = (1, 3, 3),
            stride = (1, 1, 1),
            padding = (0, 1, 1),
        )
        self.hid3 = nn.Conv3d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = (1, 3, 3),
            stride = (1, 2, 2),
            padding = (0, 1, 1),
        )
        self.hid4 = nn.Linear(32, 16)
        self.occ_out = nn.Linear(16, 1)
        # self.regression_hid = nn.Linear(32, 128)
        # self.regression_out = nn.Linear(128, 2)
    
    def tracks_from_cost_volume(
        self,
        interp_feature_heads,
        feature_grid_heads,
        query_points,
        im_shp
    ):
        cost_volume = torch.einsum(
            'bncd,bcdthw->bndthw',
            interp_feature_heads,
            feature_grid_heads,
        )
        B, N, D, T, H, W = cost_volume.shape

        cost_volume = rearrange(cost_volume, 'b n d t h w -> (b n) d t h w')

        occlusion = self.hid1(cost_volume)
        occlusion = F.relu(occlusion)

        pos = self.hid2(occlusion)
        pos = rearrange(pos, '... h w -> ... (h w)')
        pos = torch.softmax(pos * self.softmax_temperature, dim=-1)
        pos = rearrange(pos, '(b n) () t (h w) -> b n t h w', n=N, h=H, w=W)
        points = heatmaps_to_points(pos, im_shp, query_points=query_points)

        occlusion = self.hid3(occlusion)
        occlusion = torch.mean(occlusion, dim=(-1, -2)).transpose(-1, -2) # (b n) t d
        occlusion = self.hid4(occlusion)
        occlusion = F.relu(occlusion)
        occlusion = self.occ_out(occlusion)
        occlusion = occlusion.reshape(B, N, T) # bnt
        return points, occlusion

    def forward(self,
        video,
        query_points,
        compute_regression = True,
        query_chunk_size = 256,
        get_query_feats = False,
        feature_grid = None,
        query_type = 'tyx',
        **kwargs
    ):
        """
            Video: (8, 3, 24, 256, 256)
            query_points: (8, 256, 3) 3 represents (x, y, t) or (t, y, x)
            compute_regression: True
            query_chunk_size: 16
            get_query_feats: True
            feature_grid: (8, 256, 24, 32, 32)
        """
        if query_type == 'tyx':
            query_points = query_points[..., (2, 1, 0)]
        elif query_type == 'xyt':
            pass
        else:
            raise ValueError(f'Unknown query type {query_type}')
        # breakpoint()
        video = rearrange(video, 'b t h w c -> b c t h w')
        if feature_grid is None:
            latent = self.tsm_resnet(
                video,
                final_endpoint='tsm_resnet_unit_2',
            ) # (8, 256, 24, 32, 32)
            feature_grid = F.normalize(latent, dim=1, eps=1e-12)

        shape = video.shape
        # breakpoint()
        position_in_grid = convert_grid_coordinates(
            query_points,
            shape[4:1:-1],
            feature_grid.shape[4:1:-1],
        )

        position_in_grid = repeat(position_in_grid, 'b n c -> b ch n c', ch=feature_grid.shape[1]) # repeat for channel dimension
        interp_features = interp(feature_grid, position_in_grid) # b ch n

        num_heads = 1
        feature_grid_heads = rearrange(feature_grid, 'b (c d) ... -> b c d ...', d=num_heads)
        interp_features_heads = rearrange(
            interp_features,
            'b (c d) n -> b n c d',
            d=num_heads,
        )

        out = {'feature_grid': feature_grid}
        if get_query_feats:
            out['query_feats'] = interp_features


        if compute_regression:
            assert query_chunk_size is not None
            all_occ = []
            all_pts = []

            for i in range(0, query_points.shape[1], query_chunk_size):
                points, occlusion = self.tracks_from_cost_volume(
                    interp_features_heads[:, i:i + query_chunk_size],
                    feature_grid_heads,
                    query_points[:, i:i + query_chunk_size],
                    im_shp=shape[2:]
                )
                all_occ.append(occlusion)
                all_pts.append(points)
            occlusion = torch.cat(all_occ, axis=1)
            points = torch.cat(all_pts, axis=1)

            out['occlusion'] = occlusion
            out['tracks'] = points

        return out


import functools
def load_jax_tapnet(model, params_path, state_path):
    params = np.load(params_path, allow_pickle=True).item()
    state = np.load(state_path, allow_pickle=True).item()
    convert_conv = functools.partial(rearrange, pattern='t h w i o -> o i t h w')
    convert_conv2d = functools.partial(rearrange, pattern='h w i o -> o i h w')
    convert_bn = functools.partial(rearrange, pattern='() () () c -> c')
    model_state_dict = model.state_dict()

    state_dict = {
        'hid1.weight': convert_conv(params['tap_net/~/cost_volume_regression_1']['w']),
        'hid1.bias': params['tap_net/~/cost_volume_regression_1']['b'], 
        'hid2.weight': convert_conv(params['tap_net/~/cost_volume_regression_2']['w']), 
        'hid2.bias': params['tap_net/~/cost_volume_regression_2']['b'], 
        'hid3.weight': convert_conv(params['tap_net/~/cost_volume_occlusion_1']['w']),
        'hid3.bias': params['tap_net/~/cost_volume_occlusion_1']['b'],
        'hid4.weight': params['tap_net/~/cost_volume_occlusion_2']['w'].T, 
        'hid4.bias': params['tap_net/~/cost_volume_occlusion_2']['b'], 
        'occ_out.weight': params['tap_net/~/occlusion_out']['w'].T, 
        'occ_out.bias': params['tap_net/~/occlusion_out']['b'],
    }

    state_dict['tsm_resnet.conv.0.weight'] = convert_conv2d(params['tap_net/~/tsm_resnet_video/tsm_resnet_stem']['w'])
    for unit, n_blocks in zip(range(3), (2, 2, 2)):
        for block in range(n_blocks):
            if f'tsm_resnet.blocks.{unit}.blocks.{block}.projection.weight' in model_state_dict:
                state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.projection.weight'] = convert_conv2d(params[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/shortcut_conv']['w'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_0.weight'] = convert_bn(params[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm']['scale'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_0.bias'] = convert_bn(params[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm']['offset'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_0.running_var'] = convert_bn(state[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm/~/var_ema']['average'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_0.running_mean'] = convert_bn(state[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm/~/mean_ema']['average'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_0.num_batches_tracked'] = state[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm/~/mean_ema']['counter']

            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_2.weight'] = convert_bn(params[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm_1']['scale'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_2.bias'] = convert_bn(params[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm_1']['offset'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_2.running_var'] = convert_bn(state[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm_1/~/var_ema']['average'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_2.running_mean'] = convert_bn(state[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm_1/~/mean_ema']['average'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.norm_2.num_batches_tracked'] = state[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/batch_norm_1/~/mean_ema']['counter']

            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.conv_0.weight'] = convert_conv2d(params[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/conv_0']['w'])
            state_dict[f'tsm_resnet.blocks.{unit}.blocks.{block}.conv_2.weight'] = convert_conv2d(params[f'tap_net/~/tsm_resnet_video/tsm_resnet_unit_{unit}/block_{block}/conv_2']['w'])
    state_dict = {k: torch.from_numpy(v) for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def test():
    model = TAPNet()
    video = torch.rand(8, 3, 24, 256, 256)
    # feature_grid = torch.rand(8, 256, 24, 32, 32)
    feature_grid = None
    query_points = torch.rand(8, 256, 3)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True, help='Path to tapnet params.npy')
    parser.add_argument('--state', required=True, help='Path to tapnet state.npy')
    args = parser.parse_args()
    model = load_jax_tapnet(model, args.params, args.state)
    out = model(video, query_points, feature_grid=feature_grid, query_chunk_size=16, get_query_feats=True)
    print(out['tracks'].shape)
    print(out['occlusion'].shape)
    print(out['feature_grid'].shape)
    print(out['query_feats'].shape)
    torch.save(model.state_dict(), "tapnet.pt")

if __name__ == '__main__':
    test()