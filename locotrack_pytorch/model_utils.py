from typing import Sequence, Optional
import re

import numpy as np
import torch
import torch.nn.functional as F
import einops

from models.utils import convert_grid_coordinates
from data.evaluation_datasets import compute_tapvid_metrics

def huber_loss(tracks, 
               target_points, 
               occluded, 
               delta=4.0, 
               margin=2.0, 
               reduction_axes=(1, 2)):
    """
    Huber loss for point trajectories, with a margin for ignoring small errors.
    
    Args:
        tracks: [batch, time, points, 2] predicted xy positions.
        target_points: [batch, time, points, 2] ground-truth xy positions.
        occluded: [batch, time, points] 0/1 or boolean indicator for occlusion.
        delta: threshold parameter for Huber loss.
        margin: margin (in pixels) below which no penalty is applied.
        reduction_axes: dimensions across which to average the loss.
    """
    # Compute Euclidean distance
    error = tracks - target_points
    distsqr = torch.sum(error ** 2, dim=-1)
    dist = torch.sqrt(distsqr + 1e-12)  # add small eps to prevent nan

    # Shift the distance by the margin (distance below margin => 0 penalty)
    dist = F.relu(dist - margin)

    # Standard Huber loss with the shifted distance
    loss_huber = torch.where(
        dist < delta,
        0.5 * (dist ** 2),
        delta * (dist - 0.5 * delta)
    )

    # Mask out occluded points (no loss for occluded points)
    loss_huber = loss_huber * (1.0 - occluded.float())

    if reduction_axes:
        loss_huber = torch.mean(loss_huber, dim=reduction_axes)

    return loss_huber


def smooth_deadzone_l2_loss(
    tracks,
    target_points,
    occluded,
    epsilon=4.0,          # dead-zone radius (like "delta" but means "ignore within eps")
    k=10.0,               # softness; larger -> closer to hard hinge
    reduction_axes=(1, 2)
):
    """
    Smooth dead-zone L2 loss for point trajectories.

    Same inputs/outputs convention as huber_loss():
      - tracks: (..., 2) predicted points
      - target_points: (..., 2) GT points
      - occluded: (...) boolean/binary mask (1 = occluded) broadcastable to per-point loss
    Behavior:
      - For dist <= epsilon: near-zero loss/gradient
      - For dist  > epsilon: grows ~ (dist - epsilon)^2 (smoothly via softplus)
    """
    error = tracks - target_points                              # (..., 2)
    distsqr = torch.sum(error ** 2, dim=-1)                      # (...)
    dist = torch.sqrt(distsqr + 1e-12)                           # (...)

    # Smooth hinge: approx max(0, dist - epsilon)
    # softplus(k * x) / k -> max(0, x) as k -> +inf
    hinge = F.softplus(k * (dist - epsilon)) / k                 # (...)

    loss = hinge ** 2                                            # smooth dead-zone L2
    loss = loss * (1.0 - occluded.float())

    if reduction_axes:
        loss = torch.mean(loss, dim=reduction_axes)

    return loss



def prob_loss(tracks, 
              expd, 
              target_points, 
              occluded, 
              expected_dist_thresh=8.0, 
              reduction_axes=(1, 2)):
    """
    Loss for classifying if a point is within pixel threshold of its target.
    """
    err = torch.sum((tracks - target_points) ** 2, dim=-1)
    invalid = (err > expected_dist_thresh ** 2).float()
    logprob = F.binary_cross_entropy_with_logits(expd, invalid, reduction='none')
    # Note: We only exclude GT-occluded points here (unchanged).
    logprob = logprob * (1.0 - occluded.float())
    
    if reduction_axes:
        logprob = torch.mean(logprob, dim=reduction_axes)
        
    return logprob


def tapnet_loss(points, 
                occlusion, 
                target_points, 
                target_occ, 
                shape, 
                mask=None, 
                expected_dist=None,
                position_loss_weight=0.05, 
                expected_dist_thresh=6.0, 
                huber_loss_delta=4.0,
                huber_loss_margin=2.0,
                rebalance_factor=None, 
                occlusion_loss_mask=None,
                prob_loss_weight=1.0,
                occ_loss_weight=1.0,
                pred_occ=None,
                exclude_predicted_occlusion=False,
                use_smooth_deadzone_loss=False
    ):
    """
    TAPNet loss that uses a Huber loss with a margin for the position component.
    
    Args:
        points: predicted points in normalized coordinates [batch, time, points, 2].
        occlusion: predicted occlusion logits [batch, time, points].
        target_points: ground truth points in normalized coordinates.
        target_occ: occlusion GT [batch, time, points].
        shape: shape of the video (for coordinate conversion).
        mask: optional mask to scale the entire loss.
        expected_dist: predicted expected distance (used in prob_loss).
        position_loss_weight: weighting factor for the position (Huber) loss.
        expected_dist_thresh: threshold for prob_loss classification.
        huber_loss_delta: delta parameter for Huber loss.
        huber_loss_margin: margin (in pixels) to ignore small distance errors.
        rebalance_factor: optional rebalancing factor for occlusion loss.
        occlusion_loss_mask: optional mask for occlusion loss.
        prob_loss_weight: weighting factor for the probability (distance classification) loss.
        occ_loss_weight: weighting factor for the occlusion classification loss.
        pred_occ: optional boolean mask for predicted occlusion (same shape as target_occ).
        exclude_predicted_occlusion: if True, exclude predicted-occluded points
                                     from position loss in addition to GT-occluded.

    Returns:
        loss_huber, loss_occ, loss_prob:
            - Huber loss term
            - Occlusion classification loss
            - Probability loss term (if expected_dist is provided)
    """

    if mask is None:
        mask = torch.tensor(1.0)

    # Convert points from normalized to (256, 256) if necessary
    points = convert_grid_coordinates(
        points, 
        shape[3:1:-1], 
        (256, 256), 
        coordinate_format='xy'
    )
    target_points = convert_grid_coordinates(
        target_points, 
        shape[3:1:-1], 
        (256, 256), 
        coordinate_format='xy'
    )

    # If exclude_predicted_occlusion is True and we have pred_occ,
    # combine GT occlusion with predicted occlusion
    if exclude_predicted_occlusion and (pred_occ is not None):
        # Convert both to boolean, then OR them
        combined_occ = (target_occ.bool() | pred_occ.bool()).float()
    else:
        # Only use GT occlusion
        combined_occ = target_occ.float()

    # Position (Huber) loss with margin
    if use_smooth_deadzone_loss:
        loss_huber = smooth_deadzone_l2_loss(
            tracks=points,
            target_points=target_points,
            occluded=combined_occ,
            epsilon=4,
            k=10.0,
            reduction_axes=None
        )
    else:
        loss_huber = huber_loss(
            tracks=points, 
            target_points=target_points, 
            occluded=combined_occ, 
            delta=huber_loss_delta, 
            margin=huber_loss_margin,
            reduction_axes=None
        )
    loss_huber = loss_huber * mask
    loss_huber = torch.mean(loss_huber) * position_loss_weight

    # Probability loss if expected_dist is provided
    if expected_dist is None:
        loss_prob = torch.tensor(0.0, device=points.device)
    else:
        loss_prob_raw = prob_loss(
            tracks=points.detach(), 
            expd=expected_dist, 
            target_points=target_points, 
            occluded=target_occ,  # Only GT occlusion here
            expected_dist_thresh=expected_dist_thresh, 
            reduction_axes=None
        )
        loss_prob = torch.mean(loss_prob_raw * mask) * prob_loss_weight

    # Occlusion loss (uses only GT occlusion)
    target_occ = target_occ.to(dtype=occlusion.dtype)
    loss_occ_raw = F.binary_cross_entropy_with_logits(
        occlusion, 
        target_occ, 
        reduction='none'
    )
    loss_occ_raw = loss_occ_raw * mask

    # Optionally rebalance
    if rebalance_factor is not None:
        loss_occ_raw = loss_occ_raw * ((1 + rebalance_factor) - rebalance_factor * target_occ)

    # Optionally mask out occlusion loss
    if occlusion_loss_mask is not None:
        loss_occ_raw = loss_occ_raw * occlusion_loss_mask

    loss_occ = torch.mean(loss_occ_raw) * occ_loss_weight

    return loss_huber, loss_occ, loss_prob


def tapir_loss(
    batch, 
    output,
    position_loss_weight=0.05,
    expected_dist_thresh=6.0,
    huber_loss_margin=0.,
    prob_loss_weight=1.0,
    occ_loss_weight=1.0,
    exclude_predicted_occlusion=False,
    **kwargs,
):
    """
    Combine the TAPNet losses into a final scalar loss with logged components.

    Args:
        batch: dictionary containing ground-truth data (e.g. target_points, occluded).
        output: dictionary containing model outputs (e.g. tracks, occlusion, expected_dist).
        position_loss_weight: weighting factor for the position (Huber) loss.
        expected_dist_thresh: threshold for probability loss classification.
        huber_loss_margin: margin (in pixels) to ignore small distance errors in Huber loss.
        prob_loss_weight: weighting factor for probability (distance classification) loss.
        occ_loss_weight: weighting factor for the occlusion classification loss.
        exclude_predicted_occlusion: if True, exclude predicted-occluded points 
                                     from position loss in addition to GT-occluded.
    """

    # ------------------------------------------------------------------
    # 1) Optionally compute a predicted occlusion mask for main outputs
    #    if exclude_predicted_occlusion is True
    # ------------------------------------------------------------------
    pred_occ_main = None
    if exclude_predicted_occlusion:
        occlusion_logits = output['occlusion']
        pred_occ_main = torch.sigmoid(occlusion_logits)
        if 'expected_dist' in output:
            # Combine predicted occlusion probability with expected_dist 
            # to get a refined occlusion probability
            expected_dist = torch.sigmoid(output['expected_dist'])
            pred_occ_main = 1 - (1 - pred_occ_main) * (1 - expected_dist)
        # Final threshold => boolean mask
        pred_occ_main = (pred_occ_main > 0.5)

    # ------------------------------------------------------------------
    # 2) Compute main losses
    # ------------------------------------------------------------------
    loss_scalars = {}
    loss_huber, loss_occ, loss_prob = tapnet_loss(
        points=output['tracks'],
        occlusion=output['occlusion'],
        target_points=batch['target_points'],
        target_occ=batch['occluded'],
        shape=batch['video'].shape,  # pytype: disable=attribute-error  # numpy-scalars
        expected_dist=output['expected_dist'] if 'expected_dist' in output else None,
        position_loss_weight=position_loss_weight,
        expected_dist_thresh=expected_dist_thresh,
        huber_loss_margin=huber_loss_margin,
        prob_loss_weight=prob_loss_weight,
        occ_loss_weight=occ_loss_weight,
        pred_occ=pred_occ_main,
        exclude_predicted_occlusion=exclude_predicted_occlusion,
    )

    loss = loss_huber + loss_occ + loss_prob
    loss_scalars['position_loss'] = loss_huber
    loss_scalars['occlusion_loss'] = loss_occ
    if 'expected_dist' in output:
        loss_scalars['prob_loss'] = loss_prob

    # ------------------------------------------------------------------
    # 3) Handle unrefined predictions
    # ------------------------------------------------------------------
    if 'unrefined_tracks' in output:
        for l in range(len(output['unrefined_tracks'])):
            loss_huber_l, loss_occ_l, loss_prob_l = tapnet_loss(
                points=output['unrefined_tracks'][l],
                occlusion=output['unrefined_occlusion'][l],
                target_points=batch['target_points'],
                target_occ=batch['occluded'],
                shape=batch['video'].shape,  # pytype: disable=attribute-error  # numpy-scalars
                expected_dist=(
                    output['unrefined_expected_dist'][l]
                    if 'unrefined_expected_dist' in output
                    else None
                ),
                position_loss_weight=position_loss_weight,
                expected_dist_thresh=expected_dist_thresh,
                huber_loss_margin=huber_loss_margin,
                prob_loss_weight=prob_loss_weight,
                occ_loss_weight=occ_loss_weight,
                pred_occ=pred_occ_main,
                exclude_predicted_occlusion=exclude_predicted_occlusion,
            )
            loss += (loss_huber_l + loss_occ_l + loss_prob_l)
            loss_scalars[f'position_loss_{l}'] = loss_huber_l
            loss_scalars[f'occlusion_loss_{l}'] = loss_occ_l
            if 'unrefined_expected_dist' in output:
                loss_scalars[f'prob_loss_{l}'] = loss_prob_l

    loss_scalars['loss'] = loss
    return loss, loss_scalars



def eval_batch(
    batch, 
    output, 
    eval_metrics_resolution = (256, 256),
    query_first = False,
):
    query_points = batch['query_points']
    query_points = convert_grid_coordinates(
        query_points,
        (1,) + batch['video'].shape[2:4],  # (1, height, width)
        (1,) + eval_metrics_resolution,  # (1, height, width)
        coordinate_format='tyx',
    )
    gt_target_points = batch['target_points']
    gt_target_points = convert_grid_coordinates(
        gt_target_points,
        batch['video'].shape[3:1:-1],  # (width, height)
        eval_metrics_resolution[::-1],  # (width, height)
        coordinate_format='xy',
    )
    gt_occluded = batch['occluded']

    tracks = output['tracks']
    tracks = convert_grid_coordinates(
        tracks,
        batch['video'].shape[3:1:-1],  # (width, height)
        eval_metrics_resolution[::-1],  # (width, height)
        coordinate_format='xy',
    )

    occlusion_logits = output['occlusion']
    pred_occ = torch.sigmoid(occlusion_logits)
    if 'expected_dist' in output:
        expected_dist = output['expected_dist']
        pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(expected_dist))
    pred_occ = pred_occ > 0.5  # threshold

    query_mode = 'first' if query_first else 'strided'
    metrics = compute_tapvid_metrics(
        query_points=query_points.detach().cpu().numpy(),
        gt_occluded=gt_occluded.detach().cpu().numpy(),
        gt_tracks=gt_target_points.detach().cpu().numpy(),
        pred_occluded=pred_occ.detach().cpu().numpy(),
        pred_tracks=tracks.detach().cpu().numpy(),
        query_mode=query_mode,
    )

    return metrics


def huber_coordinate_loss(
    pred_points, target_points, mask, delta=1.0, pixel_size=256
):
  """Computes the Huber loss between predicted and target coordinates.

  Args:
    pred_points (*shape, 2): point coordinates predicted by the model
    target_points (*shape, 2): target point coordinates
    mask (*shape): visibility mask
    delta (float): the threshold of the Huber loss
    pixel_size (int): pixel size of the image

  Returns:
      Continuous huber loss (*shape)
  """
  pred_points = pred_points.float()
  target_points = target_points.float()
  target_points = target_points.clip(0, pixel_size - 1)
  error = pred_points - target_points
  error = error.clip(-1e8, 1e8)  # add magnitude bound to prevent nan
  distsqr = torch.sum(torch.square(error), dim=-1, keepdims=True)
  dist = torch.sqrt(distsqr + 1e-12)
  loss = torch.where(
      dist < delta,
      distsqr / 2,
      delta * (torch.abs(dist) - delta / 2),
  )
  mask = mask.float()
  loss = (loss * mask).sum() / mask.sum()
  return loss


def coordinate_softmax(logits, labels, mask, pixel_size=256):
  """Computes the softmax loss between predicted logits and target coordinates.

  Args:
    logits (*shape, n_bins x 2): marginal softmax logits for predicting x and y
      coordinates
    labels (*shape, 2): taget coordinates
    mask (*shape): visibility mask
    pixel_size (int): pixel size of the image

  Returns:
    loss (float): the softmax loss
  """
  logits = logits.float()
  labels = labels.float()
  labels -= 0.5
  labels = labels.clip(0, pixel_size - 1)
  labels = torch.round(labels).long()
  logits = einops.rearrange(logits, 'b ... c -> b c ...')
  labels = einops.rearrange(labels, 'b ... c -> b c ...')
  logits_x, logits_y = logits.chunk(2, dim=1)
  labels_x, labels_y = labels.chunk(2, dim=1)
#   print(logits_x.shape, labels_x.shape)
  loss_x = F.cross_entropy(logits_x, labels_x.squeeze(1))
  loss_y = F.cross_entropy(logits_y, labels_y.squeeze(1))
  loss = loss_x + loss_y
  mask = mask.float()
  loss = (loss * mask).sum() / mask.sum()
  return loss


def tapnext_loss_and_grad(
      batch, 
      output, 
      **kwargs,
):
  """Computes the TAPNext loss and performs backward pass on the model.

  Use the init arg `use_checkpointing=True` when constructing TAPNext to
  optimize memory; this does not have any impact on the inference speed/quality.

  Args:
    model (TAPNext):
    batch (dict): a dictionary with 4 keys: * 'video' - a float32 tensor of
      shape [batch, time, height, width, 3]; it should be mean-std normalized
      (e.g. ImageNet normalization) * 'query_points' - a float32 tensor of shape
      [batch, num_queries, 3] - queries have the form (t, x, y); where `t` - is
      in [0, time]; x is in [0, width] and y is in [0, height] * 'target_points'
      - a float32 tensor of shape [batch, num_queries, time, 2] - target points
      of the form (y, x), same ranges as query points * 'visible' - a float32
      tensor of shape [batch, num_queries, time, 1] - visibility flags (1. is
      visible and 0. is not visible)
    loss_weight (float): weight of the loss (default: 1.0)

  Returns:
    loss (float): the total loss
  """
  loss_weight = kwargs.get('loss_weight', 1.0)
  occ_loss_weight = kwargs.get('occ_loss_weight', 1.0)
  
  pred_tracks = output['tracks'] # [b, num_queries, time, 2]
  track_logits = output['track_logits'] # [b, time, num_queries, 512]
  visible_logits = output['visible_logits'] # [b, time, num_queries, 1]
  visible = (~batch['occluded']).float() # [b, num_queries, time]  

  is_human = kwargs.get('is_human', False)

  query_points = batch['query_points'] # [b, num_queries, 3 (t, x, y)]
  time_idx = torch.arange(
      pred_tracks.shape[2], device=pred_tracks.device, dtype=query_points.dtype
  ).view(1, 1, -1)
  time_mask = (time_idx >= query_points[..., 0:1]).float()
  masked_visible = visible * time_mask

  huber_loss = huber_coordinate_loss(
      pred_tracks,
      batch['target_points'],
      masked_visible[..., None],
  )
  softmax_loss = coordinate_softmax(
      track_logits,
      batch['target_points'].transpose(1, 2).flip(-1),
      masked_visible[..., None],
  )
  coordinate_loss = 0.1 * huber_loss + (0.0 if is_human else 1.0) * softmax_loss
  visible_t = visible.transpose(1, 2).unsqueeze(-1)
  time_mask_t = time_mask.transpose(1, 2).unsqueeze(-1)
  visibility_bce = F.binary_cross_entropy_with_logits(
      visible_logits, visible_t, reduction='none'
  )
  visibility_loss = occ_loss_weight * (
      (visibility_bce * time_mask_t).sum() / time_mask_t.sum().clamp_min(1.0)
  )
  loss = (coordinate_loss + visibility_loss) * loss_weight
#   breakpoint()
  return loss, {
        'loss': loss,
        'huber_loss': huber_loss,
        'softmax_loss': softmax_loss,
        'visibility_loss': visibility_loss,
  }


def get_window(coord, softmax, radius: int = 8):
  b = coord.shape[0]
  start = torch.floor(coord - radius - 0.5).int()
  start.clamp_(min=0)
  indices = start + torch.arange(radius * 2 + 1, device=softmax.device).repeat(
      b, 1
  )
  # this is to simulate one corner case of jax implementation
  shift = (indices.max(1).values - softmax.shape[1] + 1).clamp(min=0)
  indices -= shift.unsqueeze(1)
  softmax = softmax.gather(dim=1, index=indices)
  return softmax, indices + 0.5


def tracker_certainty(coord_yx, track_logits, radius=8):
  """Computes the certainty of the tracker."""
  shape = coord_yx.shape[:-1]
  coord_yx = coord_yx.flatten(0, -2)
  track_logits = track_logits.flatten(0, -2)
  # track_logits.shape == [b, 512]
  # coord_yx.shape == [b, 2]
  logits_y, logits_x = track_logits.chunk(2, dim=-1)
  track_softmax_y = F.softmax(logits_y, dim=-1)
  track_softmax_x = F.softmax(logits_x, dim=-1)
  sm_y, coord_y = get_window(coord_yx[:, 0:1], track_softmax_y)
  sm_x, coord_x = get_window(coord_yx[:, 1:2], track_softmax_x)
  sm = sm_y[..., :, None] * sm_x[..., None, :]
  grid_x, grid_y = torch.vmap(torch.meshgrid)(coord_x, coord_y)
  # grid_x.shape == [b, N, N]
  grid = torch.stack([grid_y, grid_x], dim=-1)
  in_radius = ((grid - coord_yx[:, None, None]) ** 2).sum(-1) <= (
      (radius**2) + 1e-8
  )
  return (sm * in_radius).sum(-1).sum(-1).reshape(*shape, 1)


def restore_model_from_jax_checkpoint(model, ckpt_path):
  """Restores a TAPNext model from a JAX checkpoint."""
  ckpt = {k: v for k, v in np.load(ckpt_path).items()}
  model.lin_proj.weight.data.copy_(
      torch.tensor(ckpt['backbone/embedding/kernel'][0]).permute(3, 2, 0, 1)
  )
  model.lin_proj.bias.data.copy_(torch.tensor(ckpt['backbone/embedding/bias']))
  model.mask_token.data.copy_(torch.tensor(ckpt['backbone/mask_token']))
  model.point_query_token.data.copy_(
      torch.tensor(ckpt['backbone/point_query_token'])
  )
  model.unknown_token.data.copy_(torch.tensor(ckpt['backbone/unknown_token']))
  model.image_pos_emb.data.copy_(torch.tensor(ckpt['backbone/pos_embedding']))
  model.encoder_norm.weight.data.copy_(
      torch.tensor(ckpt['backbone/Transformer/encoder_norm/scale'])
  )
  model.encoder_norm.bias.data.copy_(
      torch.tensor(ckpt['backbone/Transformer/encoder_norm/bias'])
  )
  for layer in range(12):
    # convert ssm part
    prefix = f'backbone/Transformer/encoderblock_{layer}/ssm_block'
    ssm_params = {
        key: torch.tensor(
            ckpt[
                f'{prefix}/'
                + re.sub('weight', 'kernel', re.sub(r'\.', '/', key))
            ]
        )
        for key, _ in model.blocks[layer].ssm_block.named_parameters()
    }
    for key in ssm_params:
      if 'weight' in key:
        ssm_params[key] = ssm_params[key].T
    model.blocks[layer].ssm_block.load_state_dict(ssm_params)

    # convert vit part
    vit_params = {
        re.sub(
            f'backbone/Transformer/encoderblock_{layer}/vit_block/', '', k
        ): v
        for k, v in ckpt.items()
        if f'backbone/Transformer/encoderblock_{layer}/vit_block' in k
    }
    torch_vit_params = {}
    torch_vit_params['ln_1.weight'] = vit_params['LayerNorm_0/scale']
    torch_vit_params['ln_1.bias'] = vit_params['LayerNorm_0/bias']
    torch_vit_params['ln_2.weight'] = vit_params['LayerNorm_1/scale']
    torch_vit_params['ln_2.bias'] = vit_params['LayerNorm_1/bias']
    torch_vit_params['mlp.0.weight'] = vit_params['MlpBlock_0/Dense_0/kernel'].T
    torch_vit_params['mlp.0.bias'] = vit_params['MlpBlock_0/Dense_0/bias']
    torch_vit_params['mlp.3.weight'] = vit_params['MlpBlock_0/Dense_1/kernel'].T
    torch_vit_params['mlp.3.bias'] = vit_params['MlpBlock_0/Dense_1/bias']
    torch_vit_params['self_attention.in_proj_weight'] = np.concatenate(
        [
            vit_params['MultiHeadDotProductAttention_0/query/kernel']
            .reshape(768, 768)
            .T,
            vit_params['MultiHeadDotProductAttention_0/key/kernel']
            .reshape(768, 768)
            .T,
            vit_params['MultiHeadDotProductAttention_0/value/kernel']
            .reshape(768, 768)
            .T,
        ],
        axis=0,
    )
    torch_vit_params['self_attention.in_proj_bias'] = np.concatenate([
        vit_params['MultiHeadDotProductAttention_0/query/bias'].flatten(),
        vit_params['MultiHeadDotProductAttention_0/key/bias'].flatten(),
        vit_params['MultiHeadDotProductAttention_0/value/bias'].flatten(),
    ])
    torch_vit_params['self_attention.out_proj.weight'] = (
        vit_params['MultiHeadDotProductAttention_0/out/kernel']
        .reshape(768, 768)
        .T
    )
    torch_vit_params['self_attention.out_proj.bias'] = vit_params[
        'MultiHeadDotProductAttention_0/out/bias'
    ].flatten()
    for k in torch_vit_params:
      torch_vit_params[k] = torch.tensor(np.array(torch_vit_params[k]))
    model.blocks[layer].vit_block.load_state_dict(torch_vit_params)
  model.visible_head[0].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_0/kernel'].T)
  )
  model.visible_head[0].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_0/bias'])
  )
  model.visible_head[1].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_1/scale'])
  )
  model.visible_head[1].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_1/bias'])
  )
  model.visible_head[3].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_3/kernel'].T)
  )
  model.visible_head[3].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_3/bias'])
  )
  model.visible_head[4].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_4/scale'])
  )
  model.visible_head[4].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_4/bias'])
  )
  model.visible_head[6].weight.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_6/kernel'].T)
  )
  model.visible_head[6].bias.data.copy_(
      torch.from_numpy(ckpt['visible_head/layers_6/bias'])
  )

  model.coordinate_head[0].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_0/kernel'].T)
  )
  model.coordinate_head[0].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_0/bias'])
  )
  model.coordinate_head[1].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_1/scale'])
  )
  model.coordinate_head[1].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_1/bias'])
  )
  model.coordinate_head[3].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_3/kernel'].T)
  )
  model.coordinate_head[3].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_3/bias'])
  )
  model.coordinate_head[4].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_4/scale'])
  )
  model.coordinate_head[4].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_4/bias'])
  )
  model.coordinate_head[6].weight.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_6/kernel'].T)
  )
  model.coordinate_head[6].bias.data.copy_(
      torch.from_numpy(ckpt['coordinate_head/layers_6/bias'])
  )
  return model
