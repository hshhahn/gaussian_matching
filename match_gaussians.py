from plyfile import PlyData
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gaussian:
    def __init__(self):
        self.centroids = None
        self.normals = None
        self.opacity = None
        self.scales = None
        self.rots = None
        self.sh_base = None
        self.feature = None

    def readGaussian(self, data_path):
        gs_vertex = PlyData.read(data_path)['vertex']
        
        # Load centroids [x, y, z]
        x = gs_vertex['x'].astype(np.float32)
        y = gs_vertex['y'].astype(np.float32)
        z = gs_vertex['z'].astype(np.float32)
        self.centroids = np.stack((x, y, z), axis=-1)  # [n, 3]
        
        # Load normals [nx, ny, nz]
        nx = gs_vertex['nx'].astype(np.float32)
        ny = gs_vertex['ny'].astype(np.float32)
        nz = gs_vertex['nz'].astype(np.float32)
        self.normals = np.stack((nx, ny, nz), axis=-1)
        self.normals = self.normals / (np.linalg.norm(self.normals, axis=1, keepdims=True) + 1e-9)

        # Load opacity
        self.opacity = gs_vertex['opacity'].astype(np.float32).reshape(-1, 1)

        # Load scales [sx, sy, sz]
        scale_names = [
            p.name for p in gs_vertex.properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        self.scales = np.zeros((self.centroids.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            self.scales[:, idx] = gs_vertex[attr_name].astype(np.float32)

        # Load rotations [q_0, q_1, q_2, q_3]
        rot_names = [
            p.name for p in gs_vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        self.rots = np.zeros((self.centroids.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            self.rots[:, idx] = gs_vertex[attr_name].astype(np.float32)
        self.rots = self.rots / (np.linalg.norm(self.rots, axis=1, keepdims=True) + 1e-9)

        # Load spherical harmonic base [dc_0, dc_1, dc_2]
        self.sh_base = np.zeros((self.centroids.shape[0], 3, 1))
        self.sh_base[:, 0, 0] = gs_vertex['f_dc_0'].astype(np.float32)
        self.sh_base[:, 1, 0] = gs_vertex['f_dc_1'].astype(np.float32)
        self.sh_base[:, 2, 0] = gs_vertex['f_dc_2'].astype(np.float32)
        self.sh_base = self.sh_base.reshape(-1, 3)

        return self  # Return the instance for chaining or further use

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions q1 and q2.
    q1 and q2 are arrays of shape (4,) representing [q0, q1, q2, q3].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion q = [q0, q1, q2, q3] to a 3x3 rotation matrix.
    """
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
    ])

def sample_uniform_quaternions(num_points):
    """
    Generates uniformly distributed quaternions for 3D rotations.

    Args:
        num_points (int): Number of quaternions to sample.

    Returns:
        np.ndarray: Array of shape (num_points, 4) containing quaternions [q0, q1, q2, q3].
    """
    u1 = np.random.uniform(0, 1, num_points)
    u2 = np.random.uniform(0, 1, num_points)
    u3 = np.random.uniform(0, 1, num_points)

    q0 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q1 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    quaternions = np.stack((q0, q1, q2, q3), axis=-1)
    return quaternions

def quaternion_to_rotation_matrix_torch(q):
    """Converts a (normalized) quaternion q=[q0,q1,q2,q3] to a 3x3 rotation matrix (PyTorch)."""
    q0, q1, q2, q3 = q
    return torch.tensor([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)]
    ], dtype=torch.float32)

def apply_transform(gaussian, quarternion, translation):
    """
    Applies a transformation to the Gaussian object.
    
    Args:
        gaussian (Gaussian): The Gaussian object to transform.
        quarternion (np.ndarray): The quaternion representing the rotation.
        translation (np.ndarray): The translation vector.
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quarternion)
    
    # Apply rotation and translation
    gaussian.centroids = np.dot(gaussian.centroids, rotation_matrix.T) + translation
    gaussian.rots = np.array([quaternion_multiply(quarternion, r) for r in gaussian.rots])
    gaussian.rots = np.array([normalize_quaternion(r) for r in gaussian.rots])
    
    return gaussian

def loss_fn_torch(gaussian1, gaussian2, q, t):
    q_norm = q / (q.norm() + 1e-9)
    R = quaternion_to_rotation_matrix_torch(q_norm).to(q.device)  # Move rotation matrix to GPU

    g1_centers_torch = torch.from_numpy(gaussian1.centroids).float().to(q.device)
    g1_feat_torch = torch.from_numpy(gaussian1.feature).float().to(q.device)
    g2_centers_torch = torch.from_numpy(gaussian2.centroids).float().to(q.device)
    g2_feat_torch = torch.from_numpy(gaussian2.feature).float().to(q.device)

    g1_transformed = (g1_centers_torch @ R.T) + t

    dist_matrix = torch.cdist(g1_transformed, g2_centers_torch)  # [N1, N2]
    distances, indices = dist_matrix.min(dim=1)

    geo_loss = distances.sum()
    feature_loss = torch.sum((g1_feat_torch - g2_feat_torch[indices]) ** 2)
    return geo_loss + feature_loss

# Example usage
cup1 = Gaussian()
cup1.readGaussian('cup/test/cup_0082/point_cloud.ply')

cup2 = Gaussian()
cup2.readGaussian('cup/test/cup_0096/point_cloud.ply')

# Rotate the `rots` attribute of cup2 using an input quaternion
input_rotation = np.array([0.707, 0.3, 0.2, 0])  # Example input quaternion (90 degrees around X-axis)
cup_2 = apply_transform(cup2, input_rotation, np.array([0, 0, 0]))

# Align center of Mass
cup1.centroids -= np.mean(cup1.centroids, axis=0)
cup2.centroids -= np.mean(cup2.centroids, axis=0)

#Extract features by simply concatenating 
cup1.feature = np.concatenate((cup1.opacity, cup1.scales, cup1.rots), axis=1)
cup2.feature = np.concatenate((cup2.opacity, cup2.scales, cup2.rots), axis=1)

sample_cnt = 100
sampled_quaternions = sample_uniform_quaternions(sample_cnt)
opt_iter = 1000
q_list = []
t_list = []
loss_list = []


for i in tqdm(range(sample_cnt), desc="Quaternion Samples"):
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = optim.Adam([q, t], lr=1e-2)

    cup1_temp = Gaussian()
    cup1_temp.readGaussian('cup/test/cup_0082/point_cloud.ply')
    cup1_temp = apply_transform(cup1_temp, sampled_quaternions[i], np.array([0, 0, 0]))  # This part is CPU-based
    cup1_temp.feature = np.concatenate((cup1_temp.opacity, cup1_temp.scales, cup1_temp.rots), axis=1)

    for step in range(opt_iter):
        optimizer.zero_grad()
        loss_val = loss_fn_torch(cup1_temp, cup2, q, t)
        loss_val.backward()
        optimizer.step()

    q_list.append(q.detach().cpu().numpy())
    t_list.append(t.detach().cpu().numpy())
    loss_list.append(loss_val.item())

#find minimum loss
min_loss_idx = np.argmin(loss_list)
best_q = quaternion_multiply(q_list[min_loss_idx], sampled_quaternions[min_loss_idx])
best_q = normalize_quaternion(best_q)
best_t = t_list[min_loss_idx]
final_cup_1 = apply_transform(cup1, best_q, best_t)

#save the transformed cup1 as a new PLY file (only centroids)
output_path = 'cup/test/cup_0082/point_cloud_transformed.ply'
final_cup_1.centroids = final_cup_1.centroids.astype(np.float32)
    