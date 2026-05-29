import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
import math
import copy
from gaussian.loss_utils import ssim_img, depth_propagate_normal
from gaussian.normal_utils import normal_to_q
from gaussian.general_utils import inverse_sigmoid


def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)
    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

# TODO: Ablation on random choose pixels.
def get_pointcloud_v1(tfer, c2w, gt_rgb: torch.Tensor, gt_depth: torch.Tensor, pred_accum: torch.Tensor, N_points: int):
    '''
        gt_rgb: (3, H, W)
      gt_depth: (1, H, W)
    pred_accum: (1, H, W), 第一帧pred_accum传0就行;
    N_points就传第一帧应该采样多少个点就行;
    '''
    if pred_accum is None:
        pred_accum = torch.zeros_like(gt_depth)

    # STEP 1 Choose which pixel should we add;
    H, W = gt_rgb.shape[-2], gt_rgb.shape[-1]
    rgb = gt_rgb.unsqueeze(0)
    all_valid_num = (gt_depth>0).sum()

    gt_depth_cp = gt_depth.squeeze(0) + 0.0
    gt_depth_cp[pred_accum.squeeze(0)>tfer.cfg['adc_args']['accum_thresh']] = 0
    accum_valid_num = (gt_depth_cp>0).sum()
    # Guard against frames with zero valid depth (filter_thresh=-1.0 lets even
    # blank/degenerate frames through). 0/0 -> NaN -> int(NaN) crashes.
    if all_valid_num.item() == 0:
        N_samples = 0
    else:
        N_samples = int(accum_valid_num.item() / all_valid_num.item() * N_points)

    pc_all = tfer.transform(gt_depth.squeeze(0), 'depth', 'world', pose=c2w) # (N, 3)
    N_samples = min(N_samples, pc_all.shape[0])
    
    sampled_indices = torch.randperm(pc_all.shape[0], device=pc_all.device)[:N_samples]
    xyz = pc_all[sampled_indices] # (n, 3)
    
    # STEP 2 Get their relative rgb, q;
    rgb = gt_rgb.permute(1, 2, 0)[gt_depth.squeeze(0)>0][sampled_indices] # (N, 3)
    
    q = torch.randn((N_samples, 4), device=gt_depth.device, dtype=torch.float32) # (n, 4)

    return xyz, rgb, q # (n, 3), (n, 3), (n, 4)

def get_pointcloud(tfer, c2w, gt_rgb: torch.Tensor, gt_depth: torch.Tensor, pred_accum: torch.Tensor, N_points: int):
    xyz, rgb, q = get_pointcloud_v1(tfer, c2w, gt_rgb, gt_depth, pred_accum, N_points)
    return xyz, rgb, q

def weighting_grad(gaussian_model, _current_scores, _global_scores):
    grad_weight =  (_current_scores[:, 0]/(_global_scores[:, 0]+1e-6+_current_scores[:, 0])).unsqueeze(1) # (P, 1)
    gaussian_model._xyz.grad *= grad_weight
    gaussian_model._rgb.grad *= grad_weight
    gaussian_model._scaling.grad *= grad_weight
    gaussian_model._rotation.grad *= grad_weight
    gaussian_model._opacity.grad *= grad_weight

def get_split_properties(gaussian_model, split_mask, N=3):
    '''
    split_mask: (P,)
    '''
    # Split align long axis.

    new_property_dict = {}
    
    normalized_rotation            = gaussian_model.get_property('_rotation')[split_mask] # (p, 4)
    split_mask_scale               = gaussian_model.get_property('_scaling')[split_mask] # (p, 2)
    delta_xyz_xaxis                = torch.stack([1-2*normalized_rotation[:,2]**2-2*normalized_rotation[:,3]**2,
                                                    (2*normalized_rotation[:,1]*normalized_rotation[:,2]+2*normalized_rotation[:,0]*normalized_rotation[:,3]),
                                                    (2*normalized_rotation[:,1]*normalized_rotation[:,3]-2*normalized_rotation[:,0]*normalized_rotation[:,2])], dim=-1) * split_mask_scale[:, 0].unsqueeze(-1)
    delta_xyz_yaxis                = torch.stack([(2*normalized_rotation[:,1]*normalized_rotation[:,2]-2*normalized_rotation[:,0]*normalized_rotation[:,3]),
                                                    (1-2*normalized_rotation[:,1]**2-2*normalized_rotation[:,3]**2),
                                                   (2*normalized_rotation[:,2]*normalized_rotation[:,3]+2*normalized_rotation[:,0]*normalized_rotation[:,1])], dim=-1) * split_mask_scale[:, 1].unsqueeze(-1)
    delta_xyz                      = torch.zeros_like(gaussian_model._xyz[split_mask])
    delta_xyz[split_mask_scale[:,0]>split_mask_scale[:,1]] = delta_xyz_xaxis[split_mask_scale[:,0]>split_mask_scale[:,1]]
    delta_xyz[split_mask_scale[:,0]<split_mask_scale[:,1]] = delta_xyz_yaxis[split_mask_scale[:,0]<split_mask_scale[:,1]]
    delta_xyz /= 2.0

    new_property_dict['_xyz']      = torch.concat([gaussian_model._xyz[split_mask]-delta_xyz, gaussian_model._xyz[split_mask], gaussian_model._xyz[split_mask]+delta_xyz], dim=0)
    new_property_dict['_opacity']  = inverse_sigmoid(1 - torch.sqrt(1-torch.sigmoid(gaussian_model._opacity[split_mask].repeat(N, 1))))
    
    new_property_dict['_rotation'] = gaussian_model._rotation[split_mask].repeat(N, 1)
    new_property_dict['_scaling']  = gaussian_model.activate_dict['inv_scaling'](gaussian_model.get_property('_scaling')[split_mask]/N).repeat(N, 1)
    new_property_dict['_rgb']      = gaussian_model._rgb[split_mask].repeat(N, 1)
    return new_property_dict

def get_gaussian_mask(curr_iter, stable_mask, local_scores):
    '''
    stable_mask: (P,)
    local_scores: (P, 2)
    '''
    if curr_iter < 10:
        unoptimize_mask = None
    else:
        # We only optimize Unstable Gaussians with High Error Scores.
        error_scores_median = local_scores[~stable_mask, 1].median()
        optimize_mask       = torch.bitwise_and(~stable_mask, local_scores[:, 1]>error_scores_median)
        unoptimize_mask     = ~optimize_mask
    
    return unoptimize_mask

def get_u2_minus_u1(w2c1, w2c2, means3D, tfer):
    with torch.no_grad():
        proj_vu_1 = tfer.transform(means3D, 'world', 'pixel', pose=torch.linalg.inv(w2c1))[..., :2]
        proj_vu_2 = tfer.transform(means3D, 'world', 'pixel', pose=torch.linalg.inv(w2c2))[..., :2]
        return proj_vu_2 - proj_vu_1 # (P, 2)