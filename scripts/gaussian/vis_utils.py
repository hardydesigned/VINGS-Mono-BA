import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torchvision
import numpy as np
import os
from lietorch import SE3
import yaml
import cv2
from plyfile import PlyElement, PlyData



try:
    from gaussian.normal_utils import normal_to_q
except:
    pass

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.cameras import get_camera


def create_camera_actor(is_gt=False, scale=0.1, color=(1.0, 0.0, 0.0)):
    cam_points = scale * np.array([
        [0,   0,   0],
        [-1,  -1, 1.5],
        [1,  -1, 1.5],
        [1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
                          [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]
                                              ], cam_points[cam_line[1]]
        t_vals = np.linspace(0., 1., 50)
        begin_points, end_points
        point = begin_points[None, :] * \
            (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor

def check_pcd_with_poses(pcd, poses, interp=1, scale=0.1, color=(1.0, 0.0, 0.0)):
    '''
    pcd是ply
    poses是列表，每一项是4x4的变换矩阵
    interp是间隔
    '''
    all_record = o3d.geometry.PointCloud()
    for idx in range(0,poses.shape[0],interp):
        pose = poses[idx]
        cam_actor = create_camera_actor(scale=scale, color=color)
        cam_actor.transform(pose.cpu())
        all_record += cam_actor
    
    if pcd is not None:
        all_pcd = pcd + all_record
    else:
        all_pcd = all_record
    return all_pcd


def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]

def apply_depth_colormap(
    depth,
    accumulation,
    near_plane = 2.0,
    far_plane = 6.0,
    cmap="turbo",
):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this
    colored_image = apply_colormap(depth, cmap=cmap)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image

def draw_circles(image, coordinates, radius=5, color=(0, 0, 255)):
    for coord in coordinates:
        cv2.circle(image, tuple(coord), radius, color, thickness=2)

def vis_rgbdnua(cfg, frame_id, pred_dict, gt_dict, return_image=False):

    pred_rgb      = pred_dict['rgb'].permute(1, 2, 0)
    gt_rgb        = gt_dict['rgb'].permute(1, 2, 0)
    pred_depth    = pred_dict['depth'].permute(1, 2, 0)
    gt_depth      = gt_dict['depth'].permute(1, 2, 0)
    render_normal = pred_dict['normal'].permute(1, 2, 0)
    surf_normal    = pred_dict['surf_normal'].permute(1, 2, 0)
    accum         = pred_dict['accum'].permute(1, 2, 0)
    
    surf_normal = (surf_normal + 1.) / 2
    render_normal = (render_normal + 1.) / 2

    colored_pred_depth_map = apply_depth_colormap(pred_depth, None, near_plane=None, far_plane=None)
    colored_gt_depth_map = apply_depth_colormap(gt_depth, None, near_plane=None, far_plane=None)
    colored_accum = apply_depth_colormap(accum, None, near_plane=0.0, far_plane=1.0)

    SAVE_UNCERT = False
    if SAVE_UNCERT:
        os.makedirs(os.path.join(cfg['output']['save_dir'], 'uncert'), exist_ok=True)
        torch.save(gt_dict['uncert'].permute(1, 2, 0), os.path.join(cfg['output']['save_dir'], 'uncert', f'FrameId={str(frame_id.item()).zfill(5)}.pt'))
    
    log_uncert = torch.log(gt_dict['uncert'].permute(1, 2, 0))
    # log_uncert = gt_dict['uncert'].permute(1, 2, 0)
    colored_log_uncert = apply_depth_colormap(log_uncert, None, near_plane=None, far_plane=None)

    row0 = torch.cat([gt_rgb, colored_gt_depth_map, render_normal, colored_log_uncert], dim=1)
    row1 = torch.cat([pred_rgb, colored_pred_depth_map, surf_normal, colored_accum], dim=1)

    image_to_show = torch.cat([row0, row1], dim=0)
    image_to_show = image_to_show.permute(2, 0, 1) # (H, W, C)
    # image_to_show = torch.clamp(image_to_show, 0, 1)

    torchvision.utils.save_image(image_to_show, f"{cfg['output']['save_dir']}/rgbdnua/FrameId={str(frame_id.item()).zfill(5)}.png")
    
    
    # TTD 2024/10/09
    if "optical_flow" in list(pred_dict.keys()):
        pass
        # with torch.no_grad():
        #    flow_rgb = flow_to_image(pred_dict["optical_flow"].detach().cpu())
        #    np.save(f"{cfg['output']['save_dir']}/rgbdnua/OpticalFlow_FrameId={str(frame_id.item()).zfill(5)}.npy", pred_dict["optical_flow"].detach().cpu().numpy())
        #    cv2.imwrite(f"{cfg['output']['save_dir']}/rgbdnua/OpticalFlow_FrameId={str(frame_id.item()).zfill(5)}.png", flow_rgb)
    
    
    # Draw query uv.
    if 'use_dynamic' in list(cfg.keys()) and cfg['use_dynamic']:
        query_uv = get_query_uv(gt_dict['uncert'].permute(1, 2, 0), gt_depth).cpu().numpy() # (N, 2)
        img = cv2.imread(f"{cfg['output']['save_dir']}/rgbdnua/FrameId={str(frame_id.item()).zfill(5)}.png")
        np.savetxt(f"{cfg['output']['save_dir']}/rgbdnua/FrameId={str(frame_id.item()).zfill(5)}.txt", query_uv)
        
        draw_circles(img, query_uv[:, ::-1])
        cv2.imwrite(f"{cfg['output']['save_dir']}/rgbdnua/FrameId={str(frame_id.item()).zfill(5)}.png", img)

    c2w = gt_dict['pose']
    np.savetxt(f"{cfg['output']['save_dir']}/droid_c2w/{str(frame_id.item()).zfill(8)}.txt", c2w.cpu().numpy())
    
    
    with open(cfg['output']['save_dir']+'/keyframelist.txt', 'a') as f:
        f.write(f"{gt_dict['abs_frame_idx_list']}\n")
    
    if 'debug_mode' in list(cfg.keys()) and cfg['debug_mode']:
        # Save RGBD, c2w together.
        debug_dict = {}
        debug_dict['gt_c2w']   = gt_dict['c2w'].cpu() # (4, 4)
        debug_dict['gt_rgb']   = gt_dict['rgb'].cpu() # (3, H, W)
        debug_dict['gt_depth'] = gt_dict['depth'].cpu() # (1, H, W)
        debug_dict['pred_rgb']   = pred_rgb.cpu() # (3, H, W)
        debug_dict['pred_depth'] = pred_depth.cpu() # (1, H, W)
        torch.save(debug_dict, f"{cfg['output']['save_dir']}/debug_dict/{str(frame_id.item()).zfill(5)}.pt")

    # TTD 2024/10/25
    if return_image:
        image_to_show_npy = (image_to_show.cpu().detach().permute(1,2,0).numpy()*255.0).astype(np.uint8)[..., [2,1,0]]
        return image_to_show_npy


def construct_list_of_attributes(save_mode):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    if save_mode == "3dgs" or save_mode == "sky":
        for i in range(3):
            l.append('scale_{}'.format(i))
    elif save_mode == "2dgs":
        for i in range(2):
            l.append('scale_{}'.format(i))
    else:
        assert False, "Invalid save_mode."
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def save_ply(gaussian_model, idx, save_mode='3dgs'):
    '''
    save_mode: 2dgs, 3dgs, sky, pth
    '''
    if save_mode in ['2dgs', '3dgs', 'sky']:
        C0 = 0.28209479177387814
        def RGB2SH(rgb):
            return (rgb - 0.5) / C0
        max_sh_degree = 3
        xyz = gaussian_model._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # CPU-seitig: bei grossen Szenen (Storage Manager mit Millionen Gaussians)
        # wuerde die (N, 3, 16)-Allokation auf der GPU OOM verursachen.
        fused_color = RGB2SH(gaussian_model.get_property('_rgb').detach().float().cpu())
        features = torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        f_dc = features[:,:,0:1].transpose(1, 2).numpy().reshape(features.shape[0], -1)
        f_rest = features[:,:,1:].transpose(1, 2).numpy().reshape(features.shape[0], -1)
        
        opacities = gaussian_model._opacity.detach().cpu().numpy()
        scale = gaussian_model._scaling.detach().cpu().numpy()
        if save_mode == '3dgs' or save_mode == 'sky':
            scale = np.concatenate((scale, -10 * np.ones((scale.shape[0], 1))), axis=1)
        
        rotation = gaussian_model._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(save_mode)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        '''
        scale.shape    = (N, 2) or (N, 3)
        rotation.shape = (N, 4)
        f_dc.shape     = (N, 3)
        f_rest.shape   = (N, 45)
        '''
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        save_path = os.path.join(gaussian_model.cfg['output']['save_dir'], 'ply', f'idx={idx}_{save_mode}.ply')
        PlyData([el]).write(save_path)
        # Save Intrinsic.
        intrinsic_dict = {'fu': gaussian_model.tfer.fu, 'fv': gaussian_model.tfer.fv, 
                        'cu': gaussian_model.tfer.cu, 'cv': gaussian_model.tfer.cv, 
                        'H': gaussian_model.tfer.H, 'W': gaussian_model.tfer.W}
        intrinsic_file_path = os.path.join(gaussian_model.cfg['output']['save_dir'], 'ply', 'intrinsic.yaml')
        with open(intrinsic_file_path, "w", encoding="utf-8") as file:
            yaml.dump(intrinsic_dict, file, allow_unicode=True, sort_keys=False)
    
    elif save_mode == 'pth':
        save_path = os.path.join(gaussian_model.cfg['output']['save_dir'], 'ply', f'idx={idx}_{save_mode}.pth')
        xyz = gaussian_model._xyz.detach().cpu()
        globalkf_id = gaussian_model._globalkf_id.detach().cpu()
        save_dict = {'xyz': xyz, 'globalkf_id': globalkf_id}
        torch.save(save_dict, save_path)
    else:
        assert False, "Invalid save_mode."


def load_ply(ply_path):

    plydata = PlyData.read(ply_path)

    C0 = 0.28209479177387814
    def SHDC2RGB(sh_dc):
        return sh_dc * C0 + 0.5    
    max_sh_degree = 3

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    # for idx, attr_name in enumerate(extra_f_names):
    #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    rgb = SHDC2RGB(features_dc).reshape((-1, 3))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    property_dict_npy = {
        '_xyz': xyz,
        '_rgb': rgb,
        '_opacity': opacities,
        '_scaling': scales[:, :2],
        '_rotation': rots
    }

    return property_dict_npy


def calc_psnr(pred_img, gt_img, valid_mask=None):
    '''
    valid_mask: (H, W)
    '''
    if valid_mask is None: valid_mask = torch.ones_like(pred_img[0], dtype=torch.bool)
    mse = ((((pred_img - gt_img)) ** 2)[:, valid_mask]).view(pred_img.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean()


def get_poses_gaussians(poses, pose_scale=0.04, poses_f_idx_rate=None):
    poses_pth = torch.tensor(np.array(check_pcd_with_poses(None, poses, interp=1, scale=pose_scale).points), dtype=torch.float32, device=poses.device) # (N, 3)
    N_points  = poses_pth.shape[0]
    scales          = 0.0002*torch.ones(N_points, 2, dtype=torch.float32, device=poses.device)
    
    if poses_f_idx_rate is not None:
        colors_precomp_perpose = torch.tensor(np.array(plt.cm.get_cmap('hsv')(poses_f_idx_rate.to(torch.float32).cpu().numpy()))[:, :3], device=poses.device).to(torch.float32)
        colors_precomp = colors_precomp_perpose.unsqueeze(1).repeat(1, N_points//colors_precomp_perpose.shape[0], 1).to(poses.device).reshape(-1, 3)
    else:
        colors_precomp  = torch.zeros(N_points, 3, dtype=torch.float32, device=poses.device)
        colors_precomp[:, 0] += 1.0
        
    opacity         = 0.999*torch.ones(N_points, 1, dtype=torch.float32, device=poses.device)
    rotations       = torch.zeros(N_points, 4, dtype=torch.float32, device=poses.device)
    rotations[:, 0] = 1.0
    
    poses_dict = {}
    poses_dict['means3D'] = poses_pth
    poses_dict['opacity'] = opacity
    poses_dict['scales'] = scales
    poses_dict['rotations'] = rotations
    poses_dict['colors_precomp'] = colors_precomp
    poses_dict['render_zeros'] = torch.zeros_like(scales)
    return poses_dict


def vis_map(visual_frontend, gaussian_model, return_image=False):
     
    
    if visual_frontend is not None:
        if not visual_frontend.cfg['mode'] == 'vo_nerfslam':
            count_save = visual_frontend.video.count_save + visual_frontend.video.count_save_bias
            poses = SE3(visual_frontend.video.poses_save[:count_save]).inv().matrix().contiguous()[:].detach()
            poses_f_idx = visual_frontend.video.tstamp_save[:count_save]
            if poses.shape[0] == 0:
                return

            poses = poses[::3]
            poses_f_idx = poses_f_idx[::3]
            
            
        else:
            count_save = (visual_frontend.visual_frontend.cam0_timestamps > 0).sum()
            if count_save == 0:
                return
            poses = SE3(visual_frontend.visual_frontend.cam0_T_world[:count_save]).inv().matrix().contiguous()[:].detach()
            poses_f_idx = visual_frontend.visual_frontend.cam0_timestamps[:count_save]
            
            poses = poses[::3]
            poses_f_idx = poses_f_idx[::3]
    
    
    
    bev_intrinsic_dict  = gaussian_model.cfg['vis']['bev_intrinsic_dict']
    bev_w2c             = torch.tensor(gaussian_model.cfg['vis']['bev_w2c'], dtype=torch.float32, device=gaussian_model.device)
    
    os.makedirs(f"{gaussian_model.cfg['output']['save_dir']}/map", exist_ok=True)
    
    with torch.no_grad():
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        if visual_frontend is not None:
            poses_dict = get_poses_gaussians(poses, pose_scale=gaussian_model.cfg['vis']['pose_scale'])
            screenspace_points = torch.zeros_like(torch.concat([gaussian_model._xyz, poses_dict['means3D'].to(gaussian_model._xyz.device)], dim=0), dtype=gaussian_model.dtype, requires_grad=True, device="cuda") + 0
        else:
            screenspace_points = torch.zeros_like(gaussian_model._xyz, dtype=gaussian_model.dtype, requires_grad=True, device="cuda") + 0
        
        try:
            screenspace_points.retain_grad()
        except:
            pass
        # (1) Setup raster_settings.
        
        camera = get_camera(bev_w2c, bev_intrinsic_dict)
        # bias_per_patch = torch.zeros((int(camera.height), int(camera.width)), dtype=torch.int32, device="cuda")
        pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
         
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
            bg = torch.ones(3, device=gaussian_model.device)*0.0, # 242/255
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0, # Set None here will lead to TypeError.
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            pixel_mask = pixel_mask,
            # bias_per_patch=bias_per_patch,
            # u2_minus_u1=torch.zeros_like(gaussian_model._xyz[..., :2]),
            # stable_mask=gaussian_model._stable_mask,
            # pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # (2) Render.
        means3D        = gaussian_model.get_property('_xyz') # (N, 3)
        means2D        = screenspace_points
        opacity        = gaussian_model.get_property('_opacity') # (N, 1)
        scales         = gaussian_model.get_property('_scaling') # (N, 3)
        rotations      = gaussian_model.get_property('_rotation') # (N, 4)
        colors_precomp = gaussian_model.get_property('_rgb') # (N, 3)
        render_zeros   = gaussian_model.get_property('_zeros') # (N, 2)
        
        # Concat poses_dict.
        if visual_frontend is not None:
            if visual_frontend.dataset_length is not None:
                poses_f_idx_rate = (poses_f_idx-poses_f_idx[0]) / visual_frontend.dataset_length
            else:
                poses_f_idx_rate = None
            
            poses_dict = get_poses_gaussians(poses, pose_scale=gaussian_model.cfg['vis']['pose_scale'], poses_f_idx_rate=poses_f_idx_rate)
            
            means3D        = torch.concat([means3D, poses_dict['means3D'].to(means3D.device)], dim=0)
            opacity        = torch.concat([opacity, poses_dict['opacity'].to(opacity.device)], dim=0)
            scales         = torch.concat([scales, poses_dict['scales'].to(scales.device)], dim=0)
            rotations      = torch.concat([rotations, poses_dict['rotations'].to(rotations.device)], dim=0)
            colors_precomp = torch.concat([colors_precomp, poses_dict['colors_precomp'].to(colors_precomp.device)], dim=0)
            render_zeros   = torch.concat([render_zeros, poses_dict['render_zeros'].to(render_zeros.device)], dim=0)

        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            scores    = render_zeros,
            cov3D_precomp = None
        )
        
        if not return_image:
            rendered_image = torch.clip(rendered_image, 0.0, 1.0).cpu().to(torch.float32)
            # torch.save(rendered_image,  f"{gaussian_model.cfg['output']['save_dir']}/map/FrameId={str(viz_index_max).zfill(5)}.pt")
            torchvision.utils.save_image(rendered_image,  f"{gaussian_model.cfg['output']['save_dir']}/map/FrameId={str(count_save).zfill(5)}.png")
        else:
            rendered_image = (torch.clip(rendered_image, 0.0, 1.0).permute(1,2,0).cpu()*255).numpy().astype(np.uint8)[..., [2,1,0]]
            return rendered_image            


# TTD 2024/12/05
def get_newR(oldR, axis, angle_degrees):
    """
    Creates a rotation matrix around an arbitrary axis in PyTorch
    """
    # Convert angle to radians
    theta = angle_degrees/180 * 3.14 # torch.radians(torch.tensor(angle_degrees))
    
    # Normalize the axis vector
    axis = axis / torch.linalg.norm(axis)
    
    # Compute the components of the axis vector
    a, b, c = axis[0], axis[1], axis[2]
    
    # Compute common subexpressions
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta
    
    # Create the rotation matrix using the Rodrigues' rotation formula
    R = torch.tensor([
        [cos_theta + a*a*one_minus_cos,      a*b*one_minus_cos - c*sin_theta,  a*c*one_minus_cos + b*sin_theta],
        [a*b*one_minus_cos + c*sin_theta,    cos_theta + b*b*one_minus_cos,    b*c*one_minus_cos - a*sin_theta],
        [a*c*one_minus_cos - b*sin_theta,    b*c*one_minus_cos + a*sin_theta,  cos_theta + c*c*one_minus_cos]
    ], dtype=torch.float32).to(axis.device)
    
    new_R = R @ oldR
    return new_R


def get_bev_c2w(last3_c2ws):
    '''
    Move up 2 meters and look down
    last3_c2ws.shape = (3, 4, 4)
    '''
    # STEP 1  找到垂直于三个pose的竖直向上的方向;
    # TTD 2024/12/27
    MOVEUP_METERS = 2.0 # 4.3
    # vec1 = last3_c2ws[-1, :3, 3] - last3_c2ws[-2, :3, 3]
    # vec2 = last3_c2ws[-2, :3, 3] - last3_c2ws[-3, :3, 3]
    # height_translation = torch.cross(vec1, vec2)
    # height_translation = height_translation / height_translation[-1]
    # height_translation = height_translation/(torch.linalg.norm(height_translation)) * MOVEUP_METERS
    # bev_trans = last3_c2ws[-3, :3, 3] + height_translation # (3, )
    
    # bev_trans = last3_c2ws[-1, :3, 3] + last3_c2ws[-1, :3, :3] @ torch.tensor([0, 1, 0], dtype=torch.float32, device=last3_c2ws.device) * MOVEUP_METERS
    cur_xaxis = last3_c2ws[-1, :3, :3] @ torch.tensor([1, 0, 0], dtype=torch.float32, device=last3_c2ws.device) # (3, )
    newR = get_newR(last3_c2ws[-1, :3, :3], cur_xaxis, 0)
    # bev_c2w[:3, :3] = newR
    # bev_c2w[:3, 3] = bev_trans
    
    bev_c2w = torch.eye(4, dtype=torch.float32, device=last3_c2ws.device)
    last_c2w = last3_c2ws[-3]
    bev_trans = last_c2w[:3, -1] + last_c2w[:3, :3] @ torch.tensor([0, -1, 0], dtype=torch.float32, device=last_c2w.device) * MOVEUP_METERS
    
    bev_c2w[:3, -1] = bev_trans
    bev_c2w[:3, :3] = newR # last_c2w[:3, :3]
    
    return bev_c2w


def vis_bev(visual_frontend, gaussian_model, return_image=False):
    
    if visual_frontend is not None:
        if not visual_frontend.cfg['mode'] == 'vo_nerfslam':
            count_save = visual_frontend.video.count_save + visual_frontend.video.count_save_bias
            poses = SE3(visual_frontend.video.poses_save[:count_save]).inv().matrix().contiguous()[:].detach()
            poses_f_idx = visual_frontend.video.tstamp_save[:count_save]
            if poses.shape[0] == 0:
                return

            poses = poses
            poses_f_idx = poses_f_idx
            
            
        else:
            count_save = (visual_frontend.visual_frontend.cam0_timestamps > 0).sum()
            if count_save == 0:
                return
            poses = SE3(visual_frontend.visual_frontend.cam0_T_world[:count_save]).inv().matrix().contiguous()[:].detach()
            poses_f_idx = visual_frontend.visual_frontend.cam0_timestamps[:count_save]
            
            poses = poses
            poses_f_idx = poses_f_idx
    
    
    
    bev_intrinsic_dict  = gaussian_model.cfg['vis']['bev_intrinsic_dict']
    start_bev_id = -7 if (poses_f_idx>0).sum() > 7 else -7
    end_bev_id   = start_bev_id + 3
    bev_c2w             = get_bev_c2w(poses[poses_f_idx>0][start_bev_id:end_bev_id]).to(torch.float32).to(gaussian_model.device)
    bev_w2c             = torch.inverse(bev_c2w)
    
    os.makedirs(f"{gaussian_model.cfg['output']['save_dir']}/bev", exist_ok=True)
    
    
    with torch.no_grad():
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        if visual_frontend is not None:
            poses_dict = get_poses_gaussians(poses, pose_scale=gaussian_model.cfg['vis']['pose_scale'])
            screenspace_points = torch.zeros_like(torch.concat([gaussian_model._xyz, poses_dict['means3D'].to(gaussian_model._xyz.device)], dim=0), dtype=gaussian_model.dtype, requires_grad=True, device="cuda") + 0
        else:
            screenspace_points = torch.zeros_like(gaussian_model._xyz, dtype=gaussian_model.dtype, requires_grad=True, device="cuda") + 0
        
        try:
            screenspace_points.retain_grad()
        except:
            pass
        # (1) Setup raster_settings.
        
        camera = get_camera(bev_w2c, bev_intrinsic_dict)
        # bias_per_patch = torch.zeros((int(camera.height), int(camera.width)), dtype=torch.int32, device="cuda")
        pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()
         
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=camera.tanfovx,
            tanfovy=camera.tanfovy,
            # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
            bg = torch.ones(3, device=gaussian_model.device)*0.0, # 242/255
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0, # Set None here will lead to TypeError.
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            pixel_mask = pixel_mask,
            # bias_per_patch=bias_per_patch,
            # u2_minus_u1=torch.zeros_like(gaussian_model._xyz[..., :2]),
            # stable_mask=gaussian_model._stable_mask,
            # pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # (2) Render.
        means3D        = gaussian_model.get_property('_xyz') # (N, 3)
        means2D        = screenspace_points
        opacity        = gaussian_model.get_property('_opacity') # (N, 1)
        scales         = gaussian_model.get_property('_scaling') # (N, 3)
        rotations      = gaussian_model.get_property('_rotation') # (N, 4)
        colors_precomp = gaussian_model.get_property('_rgb') # (N, 3)
        render_zeros   = gaussian_model.get_property('_zeros') # (N, 2)
        
        # # Concat poses_dict.
        # if visual_frontend is not None:
        #     if visual_frontend.dataset_length is not None:
        #         poses_f_idx_rate = (poses_f_idx-poses_f_idx[0]) / visual_frontend.dataset_length
        #     else:
        #         poses_f_idx_rate = None
            
        #     poses_dict = get_poses_gaussians(poses, pose_scale=gaussian_model.cfg['vis']['pose_scale'], poses_f_idx_rate=poses_f_idx_rate)
            
        #     means3D        = torch.concat([means3D, poses_dict['means3D'].to(means3D.device)], dim=0)
        #     opacity        = torch.concat([opacity, poses_dict['opacity'].to(opacity.device)], dim=0)
        #     scales         = torch.concat([scales, poses_dict['scales'].to(scales.device)], dim=0)
        #     rotations      = torch.concat([rotations, poses_dict['rotations'].to(rotations.device)], dim=0)
        #     colors_precomp = torch.concat([colors_precomp, poses_dict['colors_precomp'].to(colors_precomp.device)], dim=0)
        #     render_zeros   = torch.concat([render_zeros, poses_dict['render_zeros'].to(render_zeros.device)], dim=0)

        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            scores    = render_zeros,
            cov3D_precomp = None
        )
        
        if not return_image:
            rendered_image = torch.clip(rendered_image, 0.0, 1.0).cpu().to(torch.float32)
            # torch.save(rendered_image,  f"{gaussian_model.cfg['output']['save_dir']}/map/FrameId={str(viz_index_max).zfill(5)}.pt")
            torchvision.utils.save_image(rendered_image,  f"{gaussian_model.cfg['output']['save_dir']}/bev/FrameId={str(count_save).zfill(5)}.png")
        else:
            rendered_image = (torch.clip(rendered_image, 0.0, 1.0).permute(1,2,0).cpu()*255).numpy().astype(np.uint8)[..., [2,1,0]]
            return rendered_image       


 

# TTD 2024/10/09
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_vu, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_vu (np.ndarray): Flow UV image of shape [2,H,W]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_uv = flow_vu.permute(1, 2, 0)[..., [1,0]].numpy()
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)