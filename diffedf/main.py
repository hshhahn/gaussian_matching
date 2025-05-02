import os
import sys
# sys.path.append(os.path.join('diffusion_edf'))

import torch
import numpy as np
import open3d as o3d
# from diffusion_edf import se3


demo_root = 'demo/sapien_demo_5_mug_20230727/data'
# demo_root = 'demo/sapien_demo_5_bottle_20230729/data'
for ep in sorted(os.listdir(demo_root)):
    root = f'{demo_root}/{ep}'
    print(root)

    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(0.5)]

    bbox = torch.Tensor([[-0.4, 0.4], [-0.4, 0.4], [0.5, 1.3]])

    x_ref = torch.Tensor([0.06, -0.1, 0.15])

    #############################################################################################################
    # Pick
    pick_grasp_points = torch.load(f'{root}/step_0/grasp_pcd/points.pt')
    pick_grasp_colors = torch.load(f'{root}/step_0/grasp_pcd/colors.pt')
    pick_scene_points = torch.load(f'{root}/step_0/scene_pcd/points.pt')
    pick_scene_colors = torch.load(f'{root}/step_0/scene_pcd/colors.pt')
    pick_pose = torch.load(f'{root}/step_0/target_poses/poses.pt')[0].numpy()

    pcd_pick_grasp = o3d.geometry.PointCloud()
    pcd_pick_grasp.points = o3d.utility.Vector3dVector(pick_grasp_points.numpy())
    pcd_pick_grasp.colors = o3d.utility.Vector3dVector(pick_grasp_colors.numpy())
    pcd_pick_grasp.estimate_normals()
    pcd_pick_grasp.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(pick_pose[:4]), center=[0,0,0])
    pcd_pick_grasp.translate(pick_pose[4:])
    geometries.append(pcd_pick_grasp)

    pcd_pick_scene = o3d.geometry.PointCloud()
    pcd_pick_scene.points = o3d.utility.Vector3dVector(pick_scene_points.numpy())
    pcd_pick_scene.colors = o3d.utility.Vector3dVector(pick_scene_colors.numpy())
    pcd_pick_scene.estimate_normals()
    geometries.append(pcd_pick_scene)

    coord_pick = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    coord_pick.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(pick_pose[:4]), center=[0,0,0])
    coord_pick.translate(pick_pose[4:])
    geometries.append(coord_pick)

    sphere_x_ref_pick = o3d.geometry.TriangleMesh.create_sphere(0.01)
    sphere_x_ref_pick.translate(x_ref.numpy())
    sphere_x_ref_pick.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(pick_pose[:4]))
    geometries.append(sphere_x_ref_pick)
    ##############################################################################################################

    ##############################################################################################################
    # Place
    place_grasp_points = torch.load(f'{root}/step_1/grasp_pcd/points.pt')
    place_grasp_colors = torch.load(f'{root}/step_1/grasp_pcd/colors.pt')
    place_scene_points = torch.load(f'{root}/step_1/scene_pcd/points.pt')
    place_scene_colors = torch.load(f'{root}/step_1/scene_pcd/colors.pt')
    place_pose = torch.load(f'{root}/step_1/target_poses/poses.pt').numpy()[0]

    pcd_place_grasp = o3d.geometry.PointCloud()
    pcd_place_grasp.points = o3d.utility.Vector3dVector(place_grasp_points.numpy())
    pcd_place_grasp.colors = o3d.utility.Vector3dVector(place_grasp_colors.numpy())
    pcd_place_grasp.estimate_normals()
    pcd_place_grasp.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(place_pose[:4]), center=[0,0,0])
    pcd_place_grasp.translate(place_pose[4:])
    pcd_place_grasp.translate([1.5, 0, 0])
    geometries.append(pcd_place_grasp)

    pcd_place_scene = o3d.geometry.PointCloud()
    pcd_place_scene.points = o3d.utility.Vector3dVector(place_scene_points.numpy())
    pcd_place_scene.colors = o3d.utility.Vector3dVector(place_scene_colors.numpy())
    pcd_place_scene.estimate_normals()
    pcd_place_scene.translate([1.5, 0, 0])
    geometries.append(pcd_place_scene)

    coord_place = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    coord_place.rotate(o3d.geometry.get_rotation_matrix_from_quaternion(place_pose[:4]), center=[0,0,0])
    coord_place.translate(place_pose[4:])
    coord_place.translate([1.5, 0, 0])
    geometries.append(coord_place)
    ##############################################################################################################


    o3d.visualization.draw_geometries(geometries)
