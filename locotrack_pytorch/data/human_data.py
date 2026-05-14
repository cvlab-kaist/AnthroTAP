import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random
import cv2
import decord
from typing import List, Tuple, Dict, Optional
import torchvision.transforms as T
from skimage.transform import warp, AffineTransform
import mediapy as media


def apply_affine_transform_to_image(image, transform_matrix):
    tform = AffineTransform(matrix=transform_matrix)
    transformed_image = warp(image, tform.inverse,
                             order=1, mode='edge', preserve_range=True)
    return transformed_image.astype(image.dtype)

def apply_per_frame_transforms_to_video(video, transform_matrices):
    num_frames = video.shape[0]
    transformed_video = np.empty_like(video)
    for i in range(num_frames):
        frame = video[i]
        transform_matrix = transform_matrices[i]
        transformed_frame = apply_affine_transform_to_image(frame.astype(np.float32)/255.0, transform_matrix)
        transformed_video[i] = (transformed_frame * 255.0).astype(video.dtype)
    return transformed_video

def generate_smooth_affine_parameters(num_frames):
    num_keyframes = max(2, num_frames // 4)
    keyframe_indices = np.linspace(0, num_frames - 1, num_keyframes, dtype=int)

    angles = np.random.uniform(-45, 45, size=num_keyframes)
    scales_x = np.random.uniform(0.5, 3.0, size=num_keyframes)
    scales_y = np.random.uniform(0.5, 3.0, size=num_keyframes)
    shears = np.random.uniform(-5, 5, size=num_keyframes)
    translations_x = np.random.uniform(-0.5, 0.5, size=num_keyframes)
    translations_y = np.random.uniform(-0.5, 0.5, size=num_keyframes)

    angles_interp = np.interp(np.arange(num_frames), keyframe_indices, angles)
    scales_x_interp = np.interp(np.arange(num_frames), keyframe_indices, scales_x)
    scales_y_interp = np.interp(np.arange(num_frames), keyframe_indices, scales_y)
    shears_interp = np.interp(np.arange(num_frames), keyframe_indices, shears)
    translations_x_interp = np.interp(np.arange(num_frames), keyframe_indices, translations_x)
    translations_y_interp = np.interp(np.arange(num_frames), keyframe_indices, translations_y)

    parameters_list = []
    for i in range(num_frames):
        parameters = {
            'angle': angles_interp[i],
            'scale_x': scales_x_interp[i],
            'scale_y': scales_y_interp[i],
            'shear': shears_interp[i],
            'tx': translations_x_interp[i],
            'ty': translations_y_interp[i],
            'flip_horizontal': False,
            'flip_vertical': False,
        }
        parameters_list.append(parameters)
    return parameters_list

def build_affine_transform_matrix(parameters, image_shape):
    angle = parameters['angle']
    scale_x = parameters['scale_x']
    scale_y = parameters['scale_y']
    shear = parameters['shear']
    tx = parameters['tx']
    ty = parameters['ty']
    flip_horizontal = parameters['flip_horizontal']
    flip_vertical = parameters['flip_vertical']

    angle_rad = np.deg2rad(angle)
    shear_rad = np.deg2rad(shear)

    h, w = image_shape[:2]
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0

    tx_pixels = tx * w
    ty_pixels = ty * h

    T1 = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])

    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    S_her = np.array([
        [1, np.tan(shear_rad), 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    S = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])

    F = np.eye(3)
    if flip_horizontal:
        F[0, 0] = -1
    if flip_vertical:
        F[1, 1] = -1

    core_transform = F @ S @ S_her @ R

    T2 = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])

    T3 = np.array([
        [1, 0, tx_pixels],
        [0, 1, ty_pixels],
        [0, 0, 1]
    ])

    transform = T3 @ T2 @ core_transform @ T1
    return transform


def augment_video_with_smooth_transform(video_np):
    num_frames = video_np.shape[0]
    if num_frames == 0:
        return video_np, np.array([])
    image_shape = video_np.shape[1:3]

    parameters_list = generate_smooth_affine_parameters(num_frames)

    transform_matrices = []
    for params in parameters_list:
        transform_matrix = build_affine_transform_matrix(params, image_shape)
        transform_matrices.append(transform_matrix)
    transform_matrices_np = np.array(transform_matrices)

    augmented_video = apply_per_frame_transforms_to_video(video_np, transform_matrices_np)

    return augmented_video, transform_matrices_np

def apply_per_frame_transform_to_points(points_txy, transform_matrices):
    if points_txy.shape[0] == 0:
        return points_txy

    t_indices = points_txy[:, 0].astype(int)
    y_coords = points_txy[:, 1]
    x_coords = points_txy[:, 2]

    ones = np.ones_like(x_coords)
    coords = np.stack([x_coords, y_coords, ones], axis=1)

    if np.any(t_indices < 0) or np.any(t_indices >= transform_matrices.shape[0]):
         raise IndexError(f"Point frame indices are out of bounds for transform matrices (Shape: {transform_matrices.shape}). Indices: {t_indices}")
    transform_matrices_t = transform_matrices[t_indices]

    transformed_coords = np.einsum('nij,nj->ni', transform_matrices_t, coords)

    transformed_x = transformed_coords[:, 0]
    transformed_y = transformed_coords[:, 1]
    t = points_txy[:, 0]

    transformed_points = np.stack([t, transformed_y, transformed_x], axis=1)

    return transformed_points


class VideoTrackingDataset(Dataset):
    def __init__(self,
                 folder_list: List[str],
                 sequence_length: int = 24,
                 dilation: int = 1,
                 output_size: Tuple[int, int] = (256, 256),
                 num_queries: int = 256,
                 min_visible_frames: int = 1,
                 max_retries: int = 20,
                 augment: bool = True,
                 augment_geom: bool = True,
                 aug_hflip_prob: float = 0.5,
                 aug_color_jitter: Optional[List[float]] = [0.3, 0.3, 0.3, 0.1],
                 video_base_dir: Optional[str] = None,
                 ):

        super().__init__()
        self.sequence_length = sequence_length
        self.dilation = dilation
        self.output_h, self.output_w = output_size
        self.num_queries = num_queries
        self.min_visible_frames = min_visible_frames
        self.max_retries = max_retries
        self.augment = augment
        self.augment_geom = augment_geom
        self.aug_hflip_prob = aug_hflip_prob
        self.video_base_dir = video_base_dir

        self.file_paths = []
        for folder in folder_list:
            search_pattern = os.path.join(folder, '**', '*.npy')
            self.file_paths.extend(glob.glob(search_pattern, recursive=True))

        if not self.file_paths:
            print(f"Warning: No .npy files found in: {folder_list}")

        print(f"Found {len(self.file_paths)} .npy files.")

        self.color_jitter_transform = None
        if self.augment and aug_color_jitter is not None and len(aug_color_jitter) == 4:
            self.color_jitter_transform = T.ColorJitter(
                brightness=aug_color_jitter[0],
                contrast=aug_color_jitter[1],
                saturation=aug_color_jitter[2],
                hue=aug_color_jitter[3]
            )
            self.color_jitter_transform_np = T.Compose([
                T.ToPILImage(),
                self.color_jitter_transform,
                T.ToTensor(),
                T.ToPILImage()
            ])


    def __len__(self) -> int:
        return len(self.file_paths)


    def _apply_transforms_to_tracks(self, tracks_xy, transform_matrices):
        T, N, _ = tracks_xy.shape
        if N == 0: return tracks_xy

        t_coords = np.arange(T).repeat(N)

        y_coords_flat = tracks_xy[..., 1].ravel()
        x_coords_flat = tracks_xy[..., 0].ravel()

        points_txy_flat = np.stack([t_coords, y_coords_flat, x_coords_flat], axis=1)

        transformed_points_txy_flat = apply_per_frame_transform_to_points(points_txy_flat, transform_matrices)

        transformed_x_flat = transformed_points_txy_flat[:, 2]
        transformed_y_flat = transformed_points_txy_flat[:, 1]

        transformed_tracks_xy = np.stack(
            [transformed_x_flat.reshape(T, N), transformed_y_flat.reshape(T, N)],
            axis=2
        )

        return transformed_tracks_xy


    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self.file_paths: return self.__getitem__(idx + 1)

        actual_idx = idx
        npy_path = self.file_paths[actual_idx]

        for attempt in range(self.max_retries):
            try:
                try:
                    data = np.load(npy_path, allow_pickle=True).item()
                except Exception as e:
                    print(f"Warning: Error loading npy {npy_path} (attempt {attempt+1}): {e}. Trying next file if retries left.")
                    if attempt == self.max_retries - 1:
                        print(f"Failed: Could not load npy {npy_path} after {self.max_retries} attempts.")
                        return self.__getitem__(idx + 1)
                    idx = (idx + 1)
                    actual_idx = idx % len(self.file_paths)
                    npy_path = self.file_paths[actual_idx]
                    continue

                tracks_all = data['tracks']
                occlusion_all = data['occlusion']
                video_size_orig = data['video_size']
                video_path = data.get('video_path', '')

                if not video_path or not isinstance(video_path, str):
                    print(f"Warning: video_path missing or invalid in {npy_path}. Skipping.")
                    if attempt < self.max_retries - 1: continue
                    else: return self.__getitem__(idx + 1)

                if self.video_base_dir is not None:
                    relative = video_path
                    for prefix in ['/media/ssd3/datasets/lets_dance/mp4_10fps/', '/media/ssd3/datasets/lets_dance/']:
                        if relative.startswith(prefix):
                            relative = relative[len(prefix):]
                            break
                    video_path = os.path.join(self.video_base_dir, relative.lstrip('/'))
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found at constructed path: {video_path} (from {npy_path}). Skipping.")
                    if attempt < self.max_retries - 1: continue
                    else: return self.__getitem__(idx + 1)

                num_frames_total, num_tracks_all, _ = tracks_all.shape
                h_orig, w_orig = video_size_orig

                effective_length = (self.sequence_length - 1) * self.dilation + 1
                if effective_length > num_frames_total:
                    print(f"Warning: Effective sequence length {effective_length} exceeds total frames {num_frames_total} in {npy_path}. Skipping.")
                    if attempt < self.max_retries - 1: continue
                    else: return self.__getitem__(idx + 1)

                max_start_frame = num_frames_total - effective_length
                start_frame = random.randint(0, max_start_frame)
                frame_indices = np.arange(start_frame, start_frame + effective_length, self.dilation)[:self.sequence_length]

                if len(frame_indices) != self.sequence_length:
                     print(f"Warning: Frame sequence length mismatch in {npy_path}. Expected {self.sequence_length}, got {len(frame_indices)}. Skipping.")
                     if attempt < self.max_retries - 1: continue
                     else: return self.__getitem__(idx + 1)

                try:
                    vr = decord.VideoReader(video_path, ctx=decord.cpu(0), width=w_orig, height=h_orig)
                    frames_orig_np = vr.get_batch(frame_indices).asnumpy()
                except (decord.DECORDError, FileNotFoundError) as e:
                    print(f"Warning: Error loading video frames from {video_path} (attempt {attempt+1}): {e}. Trying next file.")
                    if attempt < self.max_retries - 1:
                        idx = (idx + 1); actual_idx = idx % len(self.file_paths); npy_path = self.file_paths[actual_idx]
                        continue
                    else: return self.__getitem__(idx + 1)

                sampled_tracks_raw = tracks_all[frame_indices]
                sampled_occlusion = occlusion_all[frame_indices]

                visible_in_sequence = ~sampled_occlusion
                num_visible_per_track = np.sum(visible_in_sequence, axis=0)
                valid_track_indices_initial = np.where(num_visible_per_track >= self.min_visible_frames)[0]

                if len(valid_track_indices_initial) == 0:
                    print(f"Warning: No valid tracks in {npy_path} after initial filtering. Skipping.")
                    if attempt < self.max_retries - 1: continue
                    else: return self.__getitem__(idx + 1)

                tracks_filtered_orig = sampled_tracks_raw[:, valid_track_indices_initial, :]
                occlusion_filtered = sampled_occlusion[:, valid_track_indices_initial]
                visibility_filtered = ~occlusion_filtered

                video_processed_np = frames_orig_np
                tracks_processed_orig = tracks_filtered_orig

                do_geom_augment = self.augment and self.augment_geom
                if do_geom_augment:
                    try:
                        augmented_video_orig, transform_matrices = augment_video_with_smooth_transform(frames_orig_np)
                        tracks_augmented_orig = self._apply_transforms_to_tracks(tracks_filtered_orig, transform_matrices)

                        x_aug, y_aug = tracks_augmented_orig[..., 0], tracks_augmented_orig[..., 1]
                        is_within_bounds = (x_aug >= 0) & (x_aug < w_orig) & (y_aug >= 0) & (y_aug < h_orig)

                        visibility_filtered_augmented = visibility_filtered & is_within_bounds
                        occlusion_filtered_augmented = ~visibility_filtered_augmented

                        num_visible_post_aug = np.sum(visibility_filtered_augmented, axis=0)
                        final_valid_mask = num_visible_post_aug >= self.min_visible_frames
                        final_valid_track_indices_relative = np.where(final_valid_mask)[0]

                        if len(final_valid_track_indices_relative) == 0:
                            print(f"Warning: No valid tracks after augmentation in {npy_path}. Skipping.")
                            if attempt < self.max_retries - 1: continue
                            else: return self.__getitem__(idx + 1)

                        video_processed_np = augmented_video_orig
                        tracks_processed_orig = tracks_augmented_orig[:, final_valid_track_indices_relative, :]
                        occlusion_filtered = occlusion_filtered_augmented[:, final_valid_track_indices_relative]
                        visibility_filtered = visibility_filtered_augmented[:, final_valid_track_indices_relative]

                    except Exception as aug_e:
                        pass

                if self.augment and self.color_jitter_transform is not None:
                    try:
                        jittered_video_list = []
                        for t in range(self.sequence_length):
                            frame_pil = self.color_jitter_transform_np(video_processed_np[t])
                            jittered_video_list.append(np.array(frame_pil))
                        video_processed_np = np.stack(jittered_video_list, axis=0)
                    except Exception as color_e:
                        pass

                do_hflip = self.augment and random.random() < self.aug_hflip_prob
                if do_hflip:
                    video_processed_np = np.ascontiguousarray(video_processed_np[:, :, ::-1, :])
                    tracks_processed_orig[..., 0] = w_orig - 1 - tracks_processed_orig[..., 0]

                resized_video_np = np.zeros((self.sequence_length, self.output_h, self.output_w, 3), dtype=np.uint8)
                for i in range(self.sequence_length):
                    resized_video_np[i] = cv2.resize(
                        video_processed_np[i], (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR
                    )

                scale_x = self.output_w / w_orig
                scale_y = self.output_h / h_orig
                tracks_scaled = tracks_processed_orig.copy().astype(np.float32)
                tracks_scaled[..., 0] = tracks_scaled[..., 0] * scale_x
                tracks_scaled[..., 1] = tracks_scaled[..., 1] * scale_y

                tracks_scaled[..., 0] = np.clip(tracks_scaled[..., 0], 0, self.output_w - 1)
                tracks_scaled[..., 1] = np.clip(tracks_scaled[..., 1], 0, self.output_h - 1)

                num_final_tracks = tracks_scaled.shape[1]
                time_coords = np.arange(self.sequence_length).reshape(-1, 1, 1)
                time_coords_broadcast = np.broadcast_to(time_coords, (self.sequence_length, num_final_tracks, 1))

                target_points_all_np = np.concatenate(
                    (time_coords_broadcast, tracks_scaled[..., 1:2], tracks_scaled[..., 0:1]),
                    axis=2
                ).astype(np.float32)

                visible_t_indices, visible_n_indices = np.where(visibility_filtered)
                if len(visible_t_indices) == 0:
                    print(f"Warning: No visible points found in {npy_path} after augmentation. Skipping.")
                    if attempt < self.max_retries - 1: continue
                    else: return self.__getitem__(idx + 1)

                all_visible_points = target_points_all_np[visible_t_indices, visible_n_indices, :]
                all_visible_track_indices = visible_n_indices

                total_visible_points_count = len(all_visible_points)
                if total_visible_points_count == 0:
                     print(f"Warning: No visible points found in {npy_path} after augmentation. Skipping.")
                     if attempt < self.max_retries - 1: continue
                     else: return self.__getitem__(idx + 1)

                needs_replacement = total_visible_points_count < self.num_queries
                try:
                   sampled_indices_in_visible = np.random.choice(
                       total_visible_points_count, size=self.num_queries, replace=needs_replacement
                   )
                except ValueError as e:
                    print(f"Warning: Error during query sampling: {e}. Retrying.")
                    if attempt < self.max_retries - 1: continue
                    else: return self.__getitem__(idx + 1)

                query_points_sampled_np = all_visible_points[sampled_indices_in_visible]
                sampled_track_indices_relative = all_visible_track_indices[sampled_indices_in_visible]

                target_points_queries_np = target_points_all_np[:, sampled_track_indices_relative, :]
                occluded_queries_np = occlusion_filtered[:, sampled_track_indices_relative]

                video_tensor = torch.from_numpy(resized_video_np).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0

                target_points_queries_txy = torch.from_numpy(target_points_queries_np).float().permute(1, 0, 2)

                occluded_queries = torch.from_numpy(occluded_queries_np).bool().permute(1, 0)

                query_points_final = torch.from_numpy(query_points_sampled_np).float()

                return {
                    "video": video_tensor.permute(0, 2, 3, 1),
                    "target_points": target_points_queries_txy[..., [2, 1]],
                    "occluded": occluded_queries,
                    "query_points": query_points_final
                }

            except FileNotFoundError as e:
                 print(f"Failed: Video file not found: {e}. Skipping index {actual_idx}.")
                 if attempt < self.max_retries - 1:
                     idx = (idx + 1); actual_idx = idx % len(self.file_paths); npy_path = self.file_paths[actual_idx]
                     continue
                 else: return self.__getitem__(idx + 1)
            except decord.DECORDError as e:
                 print(f"Failed: Error decoding video referenced by {npy_path}): {e}. Skipping index {actual_idx}.")
                 if attempt < self.max_retries - 1:
                     idx = (idx + 1); actual_idx = idx % len(self.file_paths); npy_path = self.file_paths[actual_idx]
                     continue
                 else: return self.__getitem__(idx + 1)
            except Exception as e:
                print(f"Failed: Unexpected error processing index {actual_idx} (npy: {npy_path}), attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt == self.max_retries - 1:
                    print(f"Failed definitively after {self.max_retries} attempts for index {actual_idx}.")
                    return self.__getitem__(idx + 1)
                else:
                    idx = (idx + 1); actual_idx = idx % len(self.file_paths); npy_path = self.file_paths[actual_idx]
                    continue

        return self.__getitem__(idx + 1)
