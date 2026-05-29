import torch
import torchvision
import os
from lietorch import SE3
import torch.nn as nn
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.cameras import get_camera
from gaussian.vis_utils import get_poses_gaussians, get_bev_c2w
import numpy as np

def tq_to_matrix(tqs: torch.Tensor):
    return SE3(tqs.cpu()).matrix()

class StorageManager: # Works in the same thread as the mapper.
    def __init__(self, cfg):
        self.cfg = cfg
        
        # float16 statt float32: halbiert den RAM-Bedarf des StorageManagers
        # (~300 MB statt ~600 MB bei 7.5M Gaussians). float16 hat ~3 signifikante
        # Stellen; für CPU-seitige Positionen (±5 cm bei 10 m) und Rotations-
        # Quaternionen ausreichend. Auf der GPU werden die Tensoren beim cpu2gpu-
        # Transfer automatisch zurück auf float32 konvertiert.
        self._xyz           = torch.empty(0, dtype=torch.float16, device='cpu')
        self._rgb           = torch.empty(0, dtype=torch.float16, device='cpu')
        self._scaling       = torch.empty(0, dtype=torch.float16, device='cpu')
        self._rotation      = torch.empty(0, dtype=torch.float16, device='cpu')
        self._opacity       = torch.empty(0, dtype=torch.float16, device='cpu')
        self._global_scores = torch.empty(0, dtype=torch.float16, device='cpu')
        self._local_scores  = torch.empty(0, dtype=torch.float16, device='cpu')
        self._stable_mask   = torch.empty(0, dtype=torch.bool,    device='cpu')

        self._globalkf_id         = torch.empty(0, dtype=torch.long,    device='cpu')
        self._globalkf_max_scores = torch.empty(0, dtype=torch.float16, device='cpu')
        self.c2ws_storage_place   = torch.empty(0, device='cpu') # Same size with global history keyframes, 0 means cpu and 1 means gpu.

        # Color Poses.
        self.dataset_length = None

        # Zähler für periodisches CPU-Pruning (analog zu storage_control auf der GPU-Seite,
        # das alle 4 Keyframes aufgerufen wird). Wird in run() inkrementiert.
        self._run_call_counter = 0

    # ------------------------------------------------------------------
    # CPU-Pruning: Pendant zu GaussianModel.storage_control()
    # ------------------------------------------------------------------

    def _apply_keep_mask(self, keep: torch.BoolTensor) -> None:
        """Wendet eine Boolean-Keep-Maske auf alle CPU-Tensoren des StorageManagers an.

        Auf der GPU-Seite übernimmt prune_tensors_from_optimizer() diese Aufgabe —
        dort muss zusätzlich der Adam-Optimizer-State (exp_avg, exp_avg_sq) synchron
        gehalten werden, weil die Gaussians nn.Parameter mit Gradientenhistorie sind.
        Hier auf der CPU gibt es keinen Optimizer und keine nn.Parameter, daher genügt
        einfaches Boolean-Indexing auf allen Tensoren.
        """
        self._xyz                 = self._xyz[keep]
        self._rgb                 = self._rgb[keep]
        self._scaling             = self._scaling[keep]
        self._rotation            = self._rotation[keep]
        self._opacity             = self._opacity[keep]
        self._global_scores       = self._global_scores[keep]
        self._local_scores        = self._local_scores[keep]
        self._stable_mask         = self._stable_mask[keep]
        self._globalkf_id         = self._globalkf_id[keep]
        self._globalkf_max_scores = self._globalkf_max_scores[keep]

    def prune_cpu_gaussians(self) -> None:
        """CPU-seitiges Pendant zu GaussianModel.storage_control().

        --- Warum _global_scores als Criterion nicht funktioniert ---
        Der ursprüngliche Ansatz nutzte _global_scores[:, 0] < 0.05 als Threshold.
        _global_scores akkumuliert jedoch über ALLE Trainingsiterationen (50 Iters/KF
        × viele KFs). Schon ein schwach sichtbarer Gaussian mit Score 1e-4/Iter hat
        nach 1000 Iterationen global_score = 0.1 — deutlich über 0.05. In der Praxis
        liegt kein einziger CPU-Gaussian unter diesem Threshold, der Prune entfernt
        daher 0 Gaussians und hilft überhaupt nicht gegen RAM-Erschöpfung.

        --- Warum sigmoid(_opacity) besser geeignet ist ---
        Auf der GPU verwaltet der Adam-Optimizer die Opacity implizit: Gaussians, die
        beim Rendering nichts beitragen, bekommen negatives Opacity-Gradient → ihr
        logit sinkt → sigmoid(logit) → 0. Dieser Mechanismus ist unabhängig von der
        Anzahl der Iterationen und liefert einen direkten, skaleninvarianten Wert:
            sigmoid(logit) ∈ (0, 1)  mit  1 = voll opak, 0 = unsichtbar

        storage_control() pruned auf der GPU Gaussians mit mittlerer Gradient-Importance
        (~0.05–0.8). Das Opacity-Kriterium erfasst eine andere, aber komplementäre
        Menge: Gaussians, die durch den Optimizer bereits als "tot" markiert wurden
        (negative logits), unabhängig davon, ob sie stabil sind oder nicht.

        Stabile Gaussians mit niedriger Opacity sind ebenfalls tote Gaussians — auf der
        GPU werden sie vom storage_control() nicht berührt (stable = immer geschützt),
        aber auf der CPU werden sie nie mehr gerendert oder trainiert. Hier macht der
        Schutz keinen Sinn, deshalb prunen wir opacity-basiert OHNE stable_mask-Gate.

        Konfiguration: storage_manager.cpu_prune_opacity_threshold (default 0.05).
        sigmoid(logit) < 0.05  ↔  logit < -2.944  → Gaussian trägt <5% Deckkraft.
        """
        if self._xyz.shape[0] == 0:
            return

        # Threshold aus Config; 0.05 = 5% Deckkraft, in der Praxis kaum sichtbar.
        threshold = self.cfg.get('storage_manager', {}).get('cpu_prune_opacity_threshold', 0.05)

        # sigmoid(_opacity) ist der aktivierte Opacity-Wert in [0,1].
        # _opacity wird als Logit gespeichert (vor Aktivierung), identisch zum Mapper.
        # squeeze(-1): _opacity hat Shape (N, 1), squeeze → (N,) für Boolean-Indexing.
        # .float(): sigmoid_cpu ist nicht fuer Half implementiert; CPU-Storage haelt
        # _opacity als float16, also vor der Aktivierung auf float32 casten.
        opacity_activated = torch.sigmoid(self._opacity.squeeze(-1).float())

        # Prune: Opacity unter Schwellwert → Gaussian ist effektiv unsichtbar.
        # Kein stable_mask-Gate: stabile unsichtbare Gaussians sind auf CPU genauso
        # nutzlos wie instabile — sie werden hier nie mehr gerendert oder trainiert.
        prune_mask = opacity_activated < threshold

        n_before = self._xyz.shape[0]
        if prune_mask.any():
            self._apply_keep_mask(~prune_mask)
        n_pruned = n_before - self._xyz.shape[0]

        print(f"[CPU-Prune] {n_pruned}/{n_before} Gaussians entfernt "
              f"(opacity < {threshold}). "
              f"Verbleibend auf CPU: {self._xyz.shape[0]}")

    def gpu2cpu(self, mapper, distance_to_cur_c2w):
        # 全都放到GPU上算了，啥时候用再在每一次跑之前丢到GPU上更新mapper;
        # 感觉这里还是按照frame_id合理些，把on_gpu_kfid对应的gaussians在cpu上删除然后替换为GPU上的就行;
        # Trim c2ws_storage_place auf same size wie distance_to_cur_c2w fuer
        # bitwise_and (analog zum cpu2gpu-Fix).
        n_kfs = distance_to_cur_c2w.shape[0]
        ongpu_kf_id_mask   = (self.c2ws_storage_place[:n_kfs] == 1)
        convey_kf_id_mask  = torch.bitwise_and(ongpu_kf_id_mask,\
                                               distance_to_cur_c2w>self.cfg['storage_manager']['distance_threshold'])
        convey_kf_id       = torch.arange(ongpu_kf_id_mask.shape[0])[convey_kf_id_mask]

        delete_gaussian_mask = torch.isin(self._globalkf_id, convey_kf_id)
        convey_gaussian_mask = torch.isin(mapper._globalkf_id, convey_kf_id.to(mapper.device))
        
        if convey_kf_id.sum() > 0:
            # Update storage manager.
            # .half(): GPU-float32 → CPU-float16 (halbiert den RAM-Bedarf).
            # _stable_mask (bool) und _globalkf_id (long) bleiben unverändert.
            self._xyz           = torch.concat((self._xyz[~delete_gaussian_mask],           mapper._xyz[convey_gaussian_mask].cpu().half()), dim=0)
            self._rgb           = torch.concat((self._rgb[~delete_gaussian_mask],           mapper._rgb[convey_gaussian_mask].cpu().half()), dim=0)
            self._scaling       = torch.concat((self._scaling[~delete_gaussian_mask],       mapper._scaling[convey_gaussian_mask].cpu().half()), dim=0)
            self._rotation      = torch.concat((self._rotation[~delete_gaussian_mask],      mapper._rotation[convey_gaussian_mask].cpu().half()), dim=0)
            self._opacity       = torch.concat((self._opacity[~delete_gaussian_mask],       mapper._opacity[convey_gaussian_mask].cpu().half()), dim=0)
            self._global_scores = torch.concat((self._global_scores[~delete_gaussian_mask], mapper._global_scores[convey_gaussian_mask].cpu().half()), dim=0)
            self._local_scores  = torch.concat((self._local_scores[~delete_gaussian_mask],  mapper._local_scores[convey_gaussian_mask].cpu().half()), dim=0)
            self._stable_mask   = torch.concat((self._stable_mask[~delete_gaussian_mask],   mapper._stable_mask[convey_gaussian_mask].cpu()), dim=0)
            self._globalkf_id         = torch.concat((self._globalkf_id[~delete_gaussian_mask],         mapper._globalkf_id[convey_gaussian_mask].cpu()), dim=0)
            self._globalkf_max_scores = torch.concat((self._globalkf_max_scores[~delete_gaussian_mask], mapper._globalkf_max_scores[convey_gaussian_mask].cpu().half()), dim=0)
            self.c2ws_storage_place[convey_kf_id] = 0

            # Update mapper.
            new_dict = mapper.prune_tensors_from_optimizer(mapper.optimizer, convey_gaussian_mask)
            mapper.update_properties(new_dict)
            mapper.update_records(mode="prune", prune_gaussianmask=convey_gaussian_mask)
            
            print(f"Convey {convey_gaussian_mask.sum().item()} Gaussians from GPU to CPU.")
            print(f"Gaussian num on GPU: {mapper._xyz.shape[0]}")
            print(f"Gaussian num on CPU: {self._xyz.shape[0]}")


    def cpu2gpu(self, mapper, distance_to_cur_c2w):
        
        near_kf_id_mask  = distance_to_cur_c2w < self.cfg['storage_manager']['distance_threshold'] # (N, )
        # Trim c2ws_storage_place auf same size wie near_kf_id_mask, falls
        # storage_place durch KF-Rejection groesser ist.
        n_kfs = near_kf_id_mask.shape[0]
        oncpu_kf_id_mask = (self.c2ws_storage_place[:n_kfs] == 0)
        convey_kf_id     = torch.arange(oncpu_kf_id_mask.shape[0])[oncpu_kf_id_mask & near_kf_id_mask]
        convey_gaussian_mask = torch.isin(self._globalkf_id, convey_kf_id)
        
        if convey_gaussian_mask.sum() > 0:
            # Update mapper.
            # .float(): CPU-float16 → float32 vor dem GPU-Transfer, weil der Mapper
            # und sein Adam-Optimizer float32 nn.Parameter erwarten.
            mapper._xyz           = nn.Parameter(torch.concat((mapper._xyz, self._xyz[convey_gaussian_mask].float().cuda())).requires_grad_(True))
            mapper._rgb           = nn.Parameter(torch.concat((mapper._rgb, self._rgb[convey_gaussian_mask].float().cuda())).requires_grad_(True))
            mapper._scaling       = nn.Parameter(torch.concat((mapper._scaling, self._scaling[convey_gaussian_mask].float().cuda())).requires_grad_(True))
            mapper._rotation      = nn.Parameter(torch.concat((mapper._rotation, self._rotation[convey_gaussian_mask].float().cuda())).requires_grad_(True))
            mapper._opacity       = nn.Parameter(torch.concat((mapper._opacity, self._opacity[convey_gaussian_mask].float().cuda())).requires_grad_(True))

            mapper._global_scores = torch.concat((mapper._global_scores, self._global_scores[convey_gaussian_mask].float().cuda()))
            mapper._local_scores  = torch.concat((mapper._local_scores,  self._local_scores[convey_gaussian_mask].float().cuda()))
            mapper._stable_mask   = torch.concat((mapper._stable_mask,   self._stable_mask[convey_gaussian_mask].cuda()))
            mapper._globalkf_id         = torch.concat((mapper._globalkf_id,         self._globalkf_id[convey_gaussian_mask].cuda()))
            mapper._globalkf_max_scores = torch.concat((mapper._globalkf_max_scores, self._globalkf_max_scores[convey_gaussian_mask].float().cuda()))
            mapper.setup_optimizer()

            self.c2ws_storage_place[convey_kf_id] = 1   
    
    
    def run(self, tracker, mapper, viz_out):
        # STEP 1 Update globalkf_c2ws.
        globalkf_c2ws = torch.linalg.inv(tq_to_matrix(tracker.video.poses_save[:viz_out['global_kf_id'][-1]])) # (N, 4, 4)
        cur_c2w       = viz_out['poses'][-1] # (4, 4)
        distance_to_cur_c2w = torch.norm(torch.matmul(torch.linalg.inv(cur_c2w).unsqueeze(0).cpu(), globalkf_c2ws.cpu())[:, :3, -1], dim=-1) # (N, ), cpu

        # FIX: distance_to_cur_c2w hat shape = global_kf_id[-1] (Slice [:N-1] aus
        # poses_save). Die c2ws_storage_place muss dieselbe shape haben fuer das
        # &-op in cpu2gpu. NUR GROW (nie shrinken) -- wenn KFs zurueckgenommen
        # werden, ist storage_place groesser; das wird in cpu2gpu via [:n_kfs]
        # gekuerzt.
        target_size = int(viz_out['global_kf_id'][-1])
        if target_size > self.c2ws_storage_place.shape[0]:
            new_added_size = target_size - self.c2ws_storage_place.shape[0]
            self.c2ws_storage_place = torch.concat(
                (self.c2ws_storage_place, torch.ones(new_added_size, dtype=torch.float32)), dim=0)

        # STEP 2
        self.cpu2gpu(mapper, distance_to_cur_c2w)

        # STEP 3
        self.gpu2cpu(mapper, distance_to_cur_c2w)

        # STEP 4  Periodisches CPU-Pruning, analog zu storage_control() auf der GPU-Seite.
        # storage_control() läuft alle 4 Keyframes; run() wird ebenfalls pro Keyframe
        # aufgerufen → gleiche Kadenz durch Modulo-4-Bedingung.
        self._run_call_counter += 1
        if self._run_call_counter % 4 == 0:
            self.prune_cpu_gaussians()
    
        
    def vis_map_storage(self, visual_frontend, mapper):
        '''
        We should rendewr a frame that contains all history gaussians.
        '''
        # STEP 0 Prepare bev render params.
        count_save = visual_frontend.video.count_save + visual_frontend.video.count_save_bias
        bev_intrinsic_dict  = mapper.cfg['vis']['bev_intrinsic_dict']
        bev_w2c             = torch.tensor(mapper.cfg['vis']['bev_w2c'], dtype=torch.float32, device=mapper.device)
        os.makedirs(f"{mapper.cfg['output']['save_dir']}/map", exist_ok=True)
        camera = get_camera(bev_w2c, bev_intrinsic_dict)
        
        if visual_frontend is not None:
            poses = SE3(visual_frontend.video.poses_save[:count_save]).inv().matrix().contiguous()[:].detach()
            poses_f_idx = visual_frontend.video.tstamp_save[:count_save]
            
            MASKTOIDX = 140
            valid_mask = poses_f_idx>0
            poses = poses[valid_mask]
            poses_f_idx = poses_f_idx[valid_mask]
            
            if poses.shape[0] == 0:
                return
        
        
        # STEP 1 Separate all history gaussians.
        FRAMES_PER_BATCH = 30 # 100
        concat_global_kf_id = torch.concat((mapper._globalkf_id.cpu(), self._globalkf_id))
        start_globalkf_id, end_globalkf_id = 0, concat_global_kf_id.max()

        num_batch = (end_globalkf_id - start_globalkf_id)//FRAMES_PER_BATCH + 1
        batch_ranges = [[batch_id*FRAMES_PER_BATCH, (batch_id+1)*FRAMES_PER_BATCH] for batch_id in range(num_batch)]
        
        # STEP 2 Render them.
        bev_rendered_list = []
        bev_accum_list    = []
        with torch.no_grad():
            for batch_id in range(len(batch_ranges)):    
                cpu_gaussian_mask = torch.bitwise_and(self._globalkf_id>=batch_ranges[batch_id][0], self._globalkf_id<=batch_ranges[batch_id][1]+5)
                gpu_gaussian_mask = torch.bitwise_and(mapper._globalkf_id>=batch_ranges[batch_id][0], mapper._globalkf_id<=batch_ranges[batch_id][1]+5)
                
                # Maybe we assum pixel_mask.shape = (H, W) in that way we make whole mask like this:
                # 0, 2, 4, 5, 8, 10, ..., 21
                # .  .  .  .  .  .  .  .  .
                # .  .  .  .  0  0  0  0  0
                # In that way we can directly use this index_bias to get half/one-third pixel id.
                # bias_per_patch = torch.zeros((int(camera.height), int(camera.width)), dtype=torch.int32, device="cuda")
                pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()    
                raster_settings = GaussianRasterizationSettings(
                            image_height=int(camera.height),
                            image_width=int(camera.width),
                            tanfovx=camera.tanfovx,
                            tanfovy=camera.tanfovy,
                            # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
                            bg = torch.ones(3, device=mapper.device)*0.0,
                            scale_modifier=1.0,
                            viewmatrix=camera.world_view_transform,
                            projmatrix=camera.full_proj_transform,
                            sh_degree=0, # Set None here will lead to TypeError.
                            campos=camera.camera_center,
                            prefiltered=False,
                            debug=False,
                            # bias_per_patch=bias_per_patch)
                            pixel_mask = pixel_mask)      
                
                means3D        = torch.concat((mapper.get_property('_xyz')[gpu_gaussian_mask], self._xyz[cpu_gaussian_mask].to(mapper.device))) # (N, 3)
                means2D        = None
                opacity        = torch.concat(( mapper.get_property('_opacity')[gpu_gaussian_mask], mapper.activate_dict['_opacity'](self._opacity[cpu_gaussian_mask].to(mapper.device)))) # (N, 1)
                scales         = torch.concat(( mapper.get_property('_scaling')[gpu_gaussian_mask], mapper.activate_dict['_scaling'](self._scaling[cpu_gaussian_mask].to(mapper.device)) )) # (N, 3)
                rotations      = torch.concat(( mapper.get_property('_rotation')[gpu_gaussian_mask], mapper.activate_dict['_rotation'](self._rotation[cpu_gaussian_mask].to(mapper.device)) )) # (N, 4)
                colors_precomp = torch.concat(( mapper.get_property('_rgb')[gpu_gaussian_mask], self._rgb[cpu_gaussian_mask].to(mapper.device) )) # (N, 3)
                render_zeros   = torch.concat(( mapper.get_property('_zeros')[gpu_gaussian_mask], torch.zeros_like(self._xyz[cpu_gaussian_mask, :2]).contiguous().to(mapper.device) )) # (N, 2) 
                
                
                # Concat poses_dict.
                if visual_frontend is not None:
                    if self.dataset_length is not None:
                        poses_f_idx_rate = (poses_f_idx-max(MASKTOIDX, poses_f_idx[0])) / self.dataset_length
                        poses_f_idx_rate = torch.clamp(poses_f_idx_rate, 0.0, 1.0).to(torch.float32) + 0.0001
                    else:
                        poses_f_idx_rate = None
                    poses_dict     = get_poses_gaussians(poses[::3], pose_scale=mapper.cfg['vis']['pose_scale'], poses_f_idx_rate=poses_f_idx_rate[::3])
                    
                    if MASKTOIDX > 0:
                        gsnumber_per_pose = poses_dict['means3D'].shape[0] // poses.shape[0]
                        poses_dict['opacity'][:((poses_f_idx_rate<0.00012).sum()*gsnumber_per_pose)] *= 0.0001
                        # poses_dict['colors_precomp'][:((poses_f_idx_rate<1e-4).sum()*gsnumber_per_pose)] *= 0.0
                        # poses_dict['colors_precomp'][:((poses_f_idx_rate<1e-4).sum()*gsnumber_per_pose)] += 1.0
                    
                    means3D        = torch.concat([means3D, poses_dict['means3D'].to(means3D.device)], dim=0)
                    opacity        = torch.concat([opacity, poses_dict['opacity'].to(opacity.device)], dim=0)
                    scales         = torch.concat([scales, poses_dict['scales'].to(scales.device)], dim=0)
                    rotations      = torch.concat([rotations, poses_dict['rotations'].to(rotations.device)], dim=0)
                    colors_precomp = torch.concat([colors_precomp, poses_dict['colors_precomp'].to(colors_precomp.device)], dim=0)
                    render_zeros   = torch.concat([render_zeros, poses_dict['render_zeros'].to(render_zeros.device)], dim=0)
                
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                
                rendered_image, _, allmap = rasterizer(
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
                
                render_alpha = allmap[1:2].to('cpu').to(torch.float32)
                bev_accum_list.append(render_alpha)
                
                rendered_image = torch.clip(rendered_image, 0.0, 1.0).to('cpu').to(torch.float32)
                bev_rendered_list.append(rendered_image)
        
                
        # STEP 3 MaxConcat all rendered images.
        final_image = torch.ones_like(bev_rendered_list[0]) * 0.0
        final_accum = torch.zeros_like(bev_accum_list[0])
        for batch_id in range(len(bev_rendered_list)):
            # update_mask = torch.bitwise_and(bev_rendered_list[batch_id] > final_image, torch.sum(bev_rendered_list[batch_id], dim=0)<2.999)
            update_mask = (bev_accum_list[batch_id] > final_accum).squeeze(0)
            final_image[:, update_mask] = bev_rendered_list[batch_id][:, update_mask]
            # TTD 2024/12/03
            final_accum[update_mask.unsqueeze(0)] = bev_accum_list[batch_id][update_mask.unsqueeze(0)].detach()
            
        torchvision.utils.save_image(final_image,  f"{mapper.cfg['output']['save_dir']}/map/FrameId={str(count_save).zfill(5)}.png")
        # TTD 2024/12/03
        # np.save(f"{mapper.cfg['output']['save_dir']}/map/Accum_FrameId={str(count_save).zfill(5)}.npy", final_accum.cpu().numpy())
    
    
    # 2024/12/02
    def vis_bev_storage(self, visual_frontend, mapper):
        '''
        We should rendewr a frame that contains all history gaussians.
        '''
        # STEP 0 Prepare bev render params.
        count_save = visual_frontend.video.count_save + visual_frontend.video.count_save_bias
        bev_intrinsic_dict  = mapper.cfg['vis']['bev_intrinsic_dict']
        # bev_w2c             = torch.tensor(mapper.cfg['vis']['bev_w2c'], dtype=torch.float32, device=mapper.device)
        
        os.makedirs(f"{mapper.cfg['output']['save_dir']}/bev", exist_ok=True)
        
        if visual_frontend is not None:
            poses = SE3(visual_frontend.video.poses_save[:count_save]).inv().matrix().contiguous()[:].detach()
            poses_f_idx = visual_frontend.video.tstamp_save[:count_save]
            
            valid_mask = poses_f_idx>0
            poses = poses[valid_mask]
            poses_f_idx = poses_f_idx[valid_mask]
            
            if poses.shape[0] == 0:
                return
        
        start_bev_id = -7 if (poses_f_idx>0).sum() > 7 else -7
        end_bev_id   = start_bev_id + 3
        bev_w2c = torch.linalg.inv(get_bev_c2w(poses[start_bev_id:end_bev_id])).to(torch.float32).to(mapper.device)
        
        camera = get_camera(bev_w2c, bev_intrinsic_dict)
        
        # STEP 1 Separate all history gaussians.
        FRAMES_PER_BATCH = 50 # 100
        concat_global_kf_id = torch.concat((mapper._globalkf_id.cpu(), self._globalkf_id))
        start_globalkf_id, end_globalkf_id = 0, concat_global_kf_id.max()

        num_batch = (end_globalkf_id - start_globalkf_id)//FRAMES_PER_BATCH + 1
        # batch_ranges = [[batch_id*FRAMES_PER_BATCH, (batch_id+1)*FRAMES_PER_BATCH] for batch_id in range(num_batch)]
        
        
        # STEP 2 Render them.
        bev_rendered_list = []
        bev_accum_list    = []
        batch_ranges = [[count_save-18, count_save]]
        
        with torch.no_grad():
            for batch_id in range(len(batch_ranges)):
                
                # 选择距离当前位姿的前5帧和后5帧
                
                cpu_gaussian_mask = torch.bitwise_and(self._globalkf_id>=batch_ranges[batch_id][0], self._globalkf_id<=batch_ranges[batch_id][1]+5)
                gpu_gaussian_mask = torch.bitwise_and(mapper._globalkf_id>=batch_ranges[batch_id][0], mapper._globalkf_id<=batch_ranges[batch_id][1]+5)
                
                # Maybe we assum pixel_mask.shape = (H, W) in that way we make whole mask like this:
                # 0, 2, 4, 5, 8, 10, ..., 21
                # .  .  .  .  .  .  .  .  .
                # .  .  .  .  0  0  0  0  0
                # In that way we can directly use this index_bias to get half/one-third pixel id.
                # bias_per_patch = torch.zeros((int(camera.height), int(camera.width)), dtype=torch.int32, device="cuda")
                pixel_mask = torch.ones(int(camera.height) * int(camera.width), dtype=torch.bool).cuda()    
                raster_settings = GaussianRasterizationSettings(
                            image_height=int(camera.height),
                            image_width=int(camera.width),
                            tanfovx=camera.tanfovx,
                            tanfovy=camera.tanfovy,
                            # bg=torch.zeros(3, device=self.device) if (self._xyz.grad is not None or random.random()>0.5) else torch.ones(3, device=self.device),
                            bg = torch.ones(3, device=mapper.device)*0.0,
                            scale_modifier=1.0,
                            viewmatrix=camera.world_view_transform,
                            projmatrix=camera.full_proj_transform,
                            sh_degree=0, # Set None here will lead to TypeError.
                            campos=camera.camera_center,
                            prefiltered=False,
                            debug=False,
                            # bias_per_patch=bias_per_patch)
                            pixel_mask = pixel_mask)      
                
                means3D        = torch.concat((mapper.get_property('_xyz')[gpu_gaussian_mask], self._xyz[cpu_gaussian_mask].to(mapper.device))) # (N, 3)
                means2D        = None
                opacity        = torch.concat(( mapper.get_property('_opacity')[gpu_gaussian_mask], mapper.activate_dict['_opacity'](self._opacity[cpu_gaussian_mask]).to(mapper.device))) # (N, 1)
                scales         = torch.concat(( mapper.get_property('_scaling')[gpu_gaussian_mask], mapper.activate_dict['_scaling'](self._scaling[cpu_gaussian_mask].to(mapper.device)) )) # (N, 3)
                rotations      = torch.concat(( mapper.get_property('_rotation')[gpu_gaussian_mask], mapper.activate_dict['_rotation'](self._rotation[cpu_gaussian_mask].to(mapper.device)) )) # (N, 4)
                colors_precomp = torch.concat(( mapper.get_property('_rgb')[gpu_gaussian_mask], self._rgb[cpu_gaussian_mask].to(mapper.device) )) # (N, 3)
                render_zeros   = torch.concat(( mapper.get_property('_zeros')[gpu_gaussian_mask], torch.zeros_like(self._xyz[cpu_gaussian_mask, :2]).contiguous().to(mapper.device) )) # (N, 2) 
                
                
                # Concat poses_dict.
                # if visual_frontend is not None:
                #     if self.dataset_length is not None:
                #         poses_f_idx_rate = (poses_f_idx-poses_f_idx[0]) / self.dataset_length
                #     else:
                #         poses_f_idx_rate = None
                #     poses_dict     = get_poses_gaussians(poses, pose_scale=mapper.cfg['vis']['pose_scale'], poses_f_idx_rate=poses_f_idx_rate)
                #     means3D        = torch.concat([means3D, poses_dict['means3D'].to(means3D.device)], dim=0)
                #     opacity        = torch.concat([opacity, poses_dict['opacity'].to(opacity.device)], dim=0)
                #     scales         = torch.concat([scales, 3*poses_dict['scales'].to(scales.device)], dim=0)
                #     rotations      = torch.concat([rotations, poses_dict['rotations'].to(rotations.device)], dim=0)
                #     colors_precomp = torch.concat([colors_precomp, poses_dict['colors_precomp'].to(colors_precomp.device)], dim=0)
                #     render_zeros   = torch.concat([render_zeros, poses_dict['render_zeros'].to(render_zeros.device)], dim=0)
                
                
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                
                rendered_image, _, allmap = rasterizer(
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
                
                render_alpha = allmap[1:2].cpu().to(torch.float32)
                bev_accum_list.append(render_alpha)
                
                rendered_image = torch.clip(rendered_image, 0.0, 1.0).cpu().to(torch.float32)
                bev_rendered_list.append(rendered_image)
        
             
        # STEP 3 MaxConcat all rendered images.
        final_image = torch.ones_like(bev_rendered_list[0]) * 0.0
        final_accum = torch.zeros_like(bev_accum_list[0])
        
        for batch_id in range(len(bev_rendered_list)):
            # update_mask = torch.bitwise_and(bev_rendered_list[batch_id] > final_image, torch.sum(bev_rendered_list[batch_id], dim=0)<2.999)
            update_mask = (bev_accum_list[batch_id] > final_accum).squeeze(0)
            final_image[:, update_mask] = bev_rendered_list[batch_id][:, update_mask]
            # TTD 2024/12/03
            final_accum[update_mask.unsqueeze(0)] = bev_accum_list[batch_id][update_mask.unsqueeze(0)].detach()
            
        torchvision.utils.save_image(final_image,  f"{mapper.cfg['output']['save_dir']}/bev/FrameId={str(count_save).zfill(5)}.png")
        # TTD 2024/12/03
        # np.save(f"{mapper.cfg['output']['save_dir']}/map/Accum_FrameId={str(count_save).zfill(5)}.npy", final_accum.cpu().numpy())
    
    
    def rectify_gaussians_storage(self, loopdetect_dict, raw_globalkf_c2ws, new_globalkf_c2ws, new_scales, loop_model, gaussian_model):
        '''
        Storage version of "rectify_gaussians".
        We don't retrain first just moving refer to relative pose?
        '''
        
        # STEP 1 Rectify gaussians' xyz.
        intrinsic = {'fv': gaussian_model.tfer.fv, 'fu': gaussian_model.tfer.fu, 'cv': gaussian_model.tfer.cv, 'cu': gaussian_model.tfer.cu, 'H': gaussian_model.tfer.H, 'W': gaussian_model.tfer.W}
        kf_id_list = torch.arange(raw_globalkf_c2ws.shape[0], dtype=torch.int32).to(raw_globalkf_c2ws.device)
        
        
        
        
        