import os
import sys
import torch
import numpy as np
import open3d as o3d
# sys.path.append(os.path.join('diffusion_edf'))
# from diffusion_edf import se3

def extract_coloured_points(
        geometries,
        colour="red",
        thresh=None):
    """
    Collect all points in `geometries` whose RGB values satisfy the
    thresholds for the selected `colour`.

    Parameters
    ----------
    colour : str
        "red"  | "green" | "custom"
    thresh : dict or None
        If `colour=="custom"`, supply a dictionary with the six keys
        {"r_min","r_max","g_min","g_max","b_min","b_max"} (values in [0,1]).
        For the predefined colours this argument is ignored.

    Returns
    -------
    o3d.geometry.PointCloud  or  None
        Point cloud containing only the selected-colour points,
        or None if no points matched.
    """

    # ---------- predefined thresholds ---------------------------------
    presets = {
        "red":   dict(r_min=0.60, r_max=1.01,
                      g_min=0.00, g_max=0.30,
                      b_min=0.00, b_max=0.30),
        "green": dict(r_min=0.00, r_max=0.30,
                      g_min=0.60, g_max=1.01,
                      b_min=0.00, b_max=0.30),
    }

    if colour in presets:
        t = presets[colour]
    elif colour == "custom" and isinstance(thresh, dict):
        required = {"r_min","r_max","g_min","g_max","b_min","b_max"}
        if not required.issubset(thresh):
            raise ValueError(f"Custom thresh must contain {required}")
        t = thresh
    else:
        raise ValueError(f"Unknown colour '{colour}'. "
                         f"Choose 'red', 'green', or 'custom'.")

    # ---------- collect coloured points -------------------------------
    sel_pts, sel_clr = [], []
    for geom in geometries:
        if not isinstance(geom, o3d.geometry.PointCloud):
            continue
        if len(geom.colors) == 0:
            continue

        clr = np.asarray(geom.colors)  # (N,3), values ∈ [0,1]
        mask = (clr[:, 0] >= t["r_min"]) & (clr[:, 0] <= t["r_max"]) & \
               (clr[:, 1] >= t["g_min"]) & (clr[:, 1] <= t["g_max"]) & \
               (clr[:, 2] >= t["b_min"]) & (clr[:, 2] <= t["b_max"])

        if mask.any():
            pts = np.asarray(geom.points)[mask]
            sel_pts.append(pts)
            sel_clr.append(clr[mask])

    if not sel_pts:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(np.concatenate(sel_pts))
    pcd.colors  = o3d.utility.Vector3dVector(np.concatenate(sel_clr))
    return pcd

def load_transform(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        quat_line = lines[0].split(':')[-1].strip()
        trans_line = lines[1].split(':')[-1].strip()

        quat = np.fromstring(quat_line.strip('[]'), sep=' ', dtype=np.float64)
        trans = np.fromstring(trans_line.strip('[]'), sep=' ', dtype=np.float64)
        return quat, trans

def quat_to_rotmat(quat):
    return o3d.geometry.get_rotation_matrix_from_quaternion(quat)

def apply_transform(pcd, R, t):
    pcd.rotate(R, center=[0, 0, 0])
    pcd.translate(t)
    return pcd

def invert_pcd_colors(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Replace every colour (r,g,b) with (1‑r, 1‑g, 1‑b).
    Works in‑place and also returns the point‑cloud for convenience.
    """
    if len(pcd.colors) == 0:
        raise ValueError("Point cloud has no colour information")
    
    clr          = np.asarray(pcd.colors)      # (N,3) floats in [0,1]
    pcd.colors   = o3d.utility.Vector3dVector(1.0 - clr)
    return pcd

red_quat, red_trans = load_transform("transform_red.txt")  # (4,), (3,)
green_quat, green_trans = load_transform("transform_green.txt")  # (4,), (3,)

demo_root = 'demo/sapien_demo_5_mug_20230727/data'

ep1 = sorted(os.listdir(demo_root))[0]
ep2 = sorted(os.listdir(demo_root))[1]

root1 = f'{demo_root}/{ep1}'
root2 = f'{demo_root}/{ep2}'

geometries1 = [o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)]
geometries2 = [o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)]



#############################################################################################################
# Pick, scene 1
pick_grasp_points = torch.load(f'{root1}/step_0/grasp_pcd/points.pt')
pick_grasp_colors = torch.load(f'{root1}/step_0/grasp_pcd/colors.pt')
pick_scene_points = torch.load(f'{root1}/step_0/scene_pcd/points.pt')
pick_scene_colors = torch.load(f'{root1}/step_0/scene_pcd/colors.pt')
pick_pose = torch.load(f'{root1}/step_0/target_poses/poses.pt')[0].numpy()

pcd_pick_grasp = o3d.geometry.PointCloud()
pcd_pick_grasp.points = o3d.utility.Vector3dVector(pick_grasp_points.numpy())
pcd_pick_grasp.colors = o3d.utility.Vector3dVector(pick_grasp_colors.numpy())
pcd_pick_grasp.estimate_normals()

# Apply original pick_pose first
pcd_pick_grasp.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(pick_pose[:4]), center=[0, 0, 0])
pcd_pick_grasp.translate(pick_pose[4:])

# Apply additional transformation from transform_red.txt
R = o3d.geometry.get_rotation_matrix_from_quaternion(green_quat)

# geometries.append(pcd_pick_grasp)

pcd_pick_scene = o3d.geometry.PointCloud()
pcd_pick_scene.points = o3d.utility.Vector3dVector(pick_scene_points.numpy())

# Colors are inverted for the demo scene
inverted_colors = 1.0 - pick_scene_colors.numpy().clip(0.0, 1.0)
pcd_pick_scene.colors = o3d.utility.Vector3dVector(inverted_colors)
pcd_pick_scene.rotate(R, center=[0, 0, 0])
pcd_pick_scene.translate(green_trans)


############################################################################
# Pick, scene 2
pick_grasp_points_2 = torch.load(f'{root2}/step_0/grasp_pcd/points.pt')
pick_grasp_colors_2 = torch.load(f'{root2}/step_0/grasp_pcd/colors.pt')
pick_scene_points_2 = torch.load(f'{root2}/step_0/scene_pcd/points.pt')
pick_scene_colors_2 = torch.load(f'{root2}/step_0/scene_pcd/colors.pt')
pick_pose_2 = torch.load(f'{root2}/step_0/target_poses/poses.pt')[0].numpy()

pcd_pick_grasp_2 = o3d.geometry.PointCloud()
pcd_pick_grasp_2.points = o3d.utility.Vector3dVector(pick_grasp_points_2.numpy())
pcd_pick_grasp_2.colors = o3d.utility.Vector3dVector(pick_grasp_colors_2.numpy())
pcd_pick_grasp_2.estimate_normals()

pcd_pick_grasp_2.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(pick_pose_2[:4]), center=[0, 0, 0])
pcd_pick_grasp_2.translate(pick_pose_2[4:])

pcd_pick_scene_2 = o3d.geometry.PointCloud()
pcd_pick_scene_2.points = o3d.utility.Vector3dVector(pick_scene_points_2.numpy())
pcd_pick_scene_2.colors = o3d.utility.Vector3dVector(pick_scene_colors_2.numpy())
pcd_pick_scene_2.estimate_normals()
# geometries.append(pcd_pick_grasp_2)
geometries1.append(pcd_pick_scene_2)

geometries2.append(pcd_pick_grasp_2)
geometries2.append(pcd_pick_scene_2)

task_cup = extract_coloured_points(geometries2, colour="red")
task_cup.paint_uniform_color([0, 1, 0])

# Step 1: Convert quaternions to rotation matrices
R_red = quat_to_rotmat(red_quat)
R_green = quat_to_rotmat(green_quat)

# Step 2: Compute inverse of red transform
R_red_inv = R_red.T  # inverse of rotation matrix is transpose
t_red_inv = -R_red_inv @ red_trans

# Step 3: Apply inverse red then green
task_cup = extract_coloured_points(geometries2, colour="red")
task_cup.paint_uniform_color([0, 1, 0])  # green

# Apply red⁻¹
task_cup.rotate(R_red_inv, center=[0, 0, 0])
task_cup.translate(t_red_inv)

# Apply green
task_cup.rotate(R_green, center=[0, 0, 0])
task_cup.translate(green_trans)

pcd_pick_grasp.rotate(R_green, center=[0, 0, 0])
pcd_pick_grasp.translate(green_trans)



geometries1.append(task_cup)
geometries1.append(pcd_pick_grasp)
# geometries1.append(pcd_pick_scene)

o3d.visualization.draw_geometries(geometries1)
