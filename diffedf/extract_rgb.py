import os
import torch
import numpy as np
import open3d as o3d

# Define the root directory for the demo
demo_root = 'demo/sapien_demo_5_mug_20230727/data'

# Get all episodes (e.g., ep0, ep1, etc.)
episodes = sorted(os.listdir(demo_root))  # Process only the first two episodes

# Function to extract red points
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

# Process the first two episodes

red_point_clouds = {}
green_point_clouds = {}

for ep in episodes:
    root = f'{demo_root}/{ep}'
    print(f"Processing {root}...")

    # Load data for the current episode
    pick_grasp_points = torch.load(f'{root}/step_0/grasp_pcd/points.pt')
    pick_grasp_colors = torch.load(f'{root}/step_0/grasp_pcd/colors.pt')
    pick_scene_points = torch.load(f'{root}/step_0/scene_pcd/points.pt')
    pick_scene_colors = torch.load(f'{root}/step_0/scene_pcd/colors.pt')
    pick_pose = torch.load(f'{root}/step_0/target_poses/poses.pt')[0].numpy()

    # Create geometries
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)]

    pcd_pick_grasp = o3d.geometry.PointCloud()
    pcd_pick_grasp.points = o3d.utility.Vector3dVector(pick_grasp_points.numpy())
    pcd_pick_grasp.colors = o3d.utility.Vector3dVector(pick_grasp_colors.numpy())
    pcd_pick_grasp.estimate_normals()
    pcd_pick_grasp.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(pick_pose[:4]), center=[0, 0, 0])
    pcd_pick_grasp.translate(pick_pose[4:])
    geometries.append(pcd_pick_grasp)

    pcd_pick_scene = o3d.geometry.PointCloud()
    pcd_pick_scene.points = o3d.utility.Vector3dVector(pick_scene_points.numpy())
    pcd_pick_scene.colors = o3d.utility.Vector3dVector(pick_scene_colors.numpy())
    pcd_pick_scene.estimate_normals()
    geometries.append(pcd_pick_scene)

    red_pcd   = extract_coloured_points(geometries, colour="red")
    green_pcd = extract_coloured_points(geometries, colour="green")

    
    if red_pcd is not None:
        print(f"Red points found for {ep}.")
        red_point_clouds[ep] = red_pcd
        
    if green_pcd is not None:
        print(f"Green points found for {ep}.")
        green_point_clouds[ep] = green_pcd

#Save
if not os.path.exists("./extracted"):
    os.makedirs("./extracted")
    
for ep, red_pcd in red_point_clouds.items():
    print(f"Extracted red point cloud for {ep}:")
    print(f"Number of points: {len(np.asarray(red_pcd.points))}")
    
    # Save the red point cloud as a .ply file
    save_path = f"./extracted/{ep}_red_points.ply"
    o3d.io.write_point_cloud(save_path, red_pcd)
    print(f"Red point cloud for {ep} saved to {save_path}.")

for ep, green_pcd in green_point_clouds.items():
    print(f"Extracted green point cloud for {ep}:")
    print(f"Number of points: {len(np.asarray(green_pcd.points))}")
    
    save_path = f"./extracted/{ep}_green_points.ply"
    o3d.io.write_point_cloud(save_path, green_pcd)
    print(f"Green point cloud for {ep} saved to {save_path}.")
    