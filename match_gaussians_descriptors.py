import numpy as np
import torch
import torch.optim as optim
from plyfile import PlyData, PlyElement
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_points_as_ply(points, ply_path):
    """
    Saves an (N,3) NumPy array of XYZ points to a PLY file.
    """
    # Define the NumPy dtype for vertices
    # Each vertex has x, y, z fields, all float32
    verts_np = np.zeros(points.shape[0],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    verts_np['x'] = points[:, 0]
    verts_np['y'] = points[:, 1]
    verts_np['z'] = points[:, 2]

    # Create a PlyElement and write
    ply_el = PlyElement.describe(verts_np, 'vertex')
    PlyData([ply_el], text=False).write(ply_path)
    

class Gaussian:
    """
    Stores Gaussian data (centroids, normals, opacity, scales, rotations, features)
    directly as PyTorch tensors.
    """
    def __init__(self):
        self.centroids = None
        self.normals = None
        self.opacity = None
        self.scales = None
        self.rots = None
        self.sh_base = None
        self.feature = None

    def readGaussian(self, data_path):
        """
        Reads the .ply file and stores all relevant attributes as PyTorch tensors.
        """
        gs_vertex = PlyData.read(data_path)['vertex']

        # Centroids [x, y, z]
        x = torch.from_numpy(gs_vertex['x'].astype(np.float32))
        y = torch.from_numpy(gs_vertex['y'].astype(np.float32))
        z = torch.from_numpy(gs_vertex['z'].astype(np.float32))
        self.centroids = torch.stack([x, y, z], dim=-1).to(device)

        # Normals [nx, ny, nz]
        nx = torch.from_numpy(gs_vertex['nx'].astype(np.float32))
        ny = torch.from_numpy(gs_vertex['ny'].astype(np.float32))
        nz = torch.from_numpy(gs_vertex['nz'].astype(np.float32))
        normals = torch.stack([nx, ny, nz], dim=-1)
        self.normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-9)
        self.normals = self.normals.to(device)

        # Opacity
        self.opacity = torch.from_numpy(gs_vertex['opacity'].astype(np.float32)).view(-1, 1).to(device)

        # Scales [sx, sy, sz]
        scale_names = [p.name for p in gs_vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scale_list = []
        for attr_name in scale_names:
            scale_list.append(torch.from_numpy(gs_vertex[attr_name].astype(np.float32)))
        self.scales = torch.stack(scale_list, dim=-1).to(device)

        # Rotations [q_0, q_1, q_2, q_3]
        rot_names = [p.name for p in gs_vertex.properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rot_list = []
        for attr_name in rot_names:
            rot_list.append(torch.from_numpy(gs_vertex[attr_name].astype(np.float32)))
        rots = torch.stack(rot_list, dim=-1)
        rots = rots / (rots.norm(dim=1, keepdim=True) + 1e-9)
        self.rots = rots.to(device)

        # Spherical harmonic base [f_dc_0, f_dc_1, f_dc_2]
        dc0 = torch.from_numpy(gs_vertex['f_dc_0'].astype(np.float32))
        dc1 = torch.from_numpy(gs_vertex['f_dc_1'].astype(np.float32))
        dc2 = torch.from_numpy(gs_vertex['f_dc_2'].astype(np.float32))
        self.sh_base = torch.stack([dc0, dc1, dc2], dim=-1).to(device)

        return self

    def __getitem__(self, index):
        return {
            'centroids': self.centroids[index],
            'opacity': self.opacity[index],
            'scales': self.scales[index],
            'rots': self.rots[index],
            'normals': self.normals[index],
            'sh_base': self.sh_base[index],
            'feature': self.feature[index]
        }


def quaternion_multiply_torch(q1, q2):
    """
    q1, q2: [..., 4] – batched quaternions in PyTorch (w,x,y,z).
    Returns quaternion multiplication result of shape [..., 4].
    """
    # q1 = (w1, x1, y1, z1), q2 = (w2, x2, y2, z2)
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

def quaternion_to_rotation_matrix_torch(q):
    """
    Converts a normalized quaternion q=[q0,q1,q2,q3] to a 3x3 rotation matrix.
    q can be shape (4,) or (N,4). Returns shape (3,3) or (N,3,3).
    """
    # Ensure q is normalized
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)

    # If we have a single quaternion shape: (4, )
    # we expand dims to unify logic
    single_input = False
    if q.dim() == 1:
        single_input = True
        q = q.unsqueeze(0)

    q0, q1, q2, q3 = torch.unbind(q, dim=-1)

    # Precompute repeated terms
    two_q0q1 = 2.0 * (q0 * q1)
    two_q0q2 = 2.0 * (q0 * q2)
    two_q0q3 = 2.0 * (q0 * q3)
    two_q1q2 = 2.0 * (q1 * q2)
    two_q1q3 = 2.0 * (q1 * q3)
    two_q2q3 = 2.0 * (q2 * q3)

    # Rotation matrix
    # row-wise expansions for batch dimension
    R = torch.stack([
        1 - 2*(q2**2 + q3**2),
        two_q1q2 - two_q0q3,
        two_q1q3 + two_q0q2,
        
        two_q1q2 + two_q0q3,
        1 - 2*(q1**2 + q3**2),
        two_q2q3 - two_q0q1,
        
        two_q1q3 - two_q0q2,
        two_q2q3 + two_q0q1,
        1 - 2*(q1**2 + q2**2)
    ], dim=-1)
    R = R.reshape(-1, 3, 3)

    if single_input:
        R = R.squeeze(0)  # Return (3,3) if input was (4,)
    return R

def sample_uniform_quaternions_torch(num_points, device='cpu'):
    """
    Generates uniformly distributed random quaternions for 3D rotations in a pure PyTorch way.
    Returns shape (num_points, 4) with (w, x, y, z).
    """
    u1 = torch.rand(num_points, device=device)
    u2 = torch.rand(num_points, device=device)
    u3 = torch.rand(num_points, device=device)

    q0 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
    q1 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
    q2 = torch.sqrt(u1)       * torch.sin(2 * np.pi * u3)
    q3 = torch.sqrt(u1)       * torch.cos(2 * np.pi * u3)

    quaternions = torch.stack((q0, q1, q2, q3), dim=-1)
    return quaternions

def apply_transform_torch(centroids, rots, q, t):
    """
    centroids: [N, 3] – positions
    rots: [N, 4] – quaternion per Gaussian
    q: [4] – global rotation quaternion
    t: [3] – global translation vector
    Returns:
        new_centroids, new_rots
    """
    # Convert q to rotation matrix
    R = quaternion_to_rotation_matrix_torch(q)  # shape: (3,3)

    # Transform centroids
    # centroids shape: (N,3), R shape: (3,3)
    new_centroids = centroids @ R.t() + t

    # Rotate each local quaternion rots by q
    # rots shape: (N,4). We need to do elementwise quaternion multiplication:
    # new_rots[i] = q * rots[i]
    # For batch multiply, we can expand q to shape (N,4).
    q_batched = q.unsqueeze(0).expand_as(rots)  # shape: (N,4)
    new_rots = quaternion_multiply_torch(q_batched, rots)
    # Re-normalize each quaternion
    new_rots = new_rots / (new_rots.norm(dim=-1, keepdim=True) + 1e-9)

    return new_centroids, new_rots

def loss_fn_torch(gaussian1, gaussian2, q, t):
    
    """
    gaussian1, gaussian2 are Gaussian objects with
    gaussian1.centroids, gaussian1.feature as Tensors on device
    q: [4], t: [3] – the parameters being optimized
    """
    # Normalize q
    q_norm = q / (q.norm() + 1e-9)
    R = quaternion_to_rotation_matrix_torch(q_norm)  # (3,3)

    # Transform gaussian1's centroids
    g1_transformed = gaussian1.centroids @ R.t() + t

    # Pairwise distances [N1, N2]
    dist_matrix = torch.cdist(g1_transformed, gaussian2.centroids)
    distances, indices = dist_matrix.min(dim=1)

    # Geometry loss
    geo_loss = distances.sum()

    # Feature loss
    g1_feat = gaussian1.feature
    g2_feat = gaussian2.feature[indices]  # gather nearest features
    lambda_ = 1
    feat_loss = torch.sum((g1_feat - g2_feat)**2)

    return geo_loss + lambda_ * feat_loss

def get_gaussian_value(point, gaussian):
    """
    point: [3] – point in 3D space
    gaussian - single gaussian
    Returns the value of the Gaussian at the given point.
    """
    # Compute distance from point to each centroid
    distances = torch.norm(point - gaussian.centroids, dim=1)
    
    # Compute Gaussian value using opacity and scales
    values = gaussian.opacity * torch.exp(-distances / (2 * gaussian.scales**2))
    
    return values

def gaussian_descriptor_field(gaussian1, gaussian2, nearest_k=5):
    # For each point in gaussian1, find k nearest points in gaussian2
    dist_matrix = torch.cdist(gaussian1.centroids, gaussian2.centroids)  # Shape: [N1, N2]

    # Find the indices of the k-nearest neighbors in gaussian2 for each point in gaussian1
    _, knn_indices = torch.topk(dist_matrix, nearest_k, dim=1, largest=False)  # Shape: [N1, k]

    # Gather the features of the k-nearest neighbors
    descriptor_field = torch.gather(gaussian2.feature, 0, knn_indices.unsqueeze(-1).expand(-1, -1, gaussian2.feature.size(1)))
    


########################### Main Code #############################
# Align obj 1 into obj 2 (Find Transform)

dataset_dir = 'dataset/modelnet_splat'
type_dir = 'guitar/test'

obj1_path = f'{dataset_dir}/{type_dir}/guitar_0156'
obj2_path = f'{dataset_dir}/{type_dir}/guitar_0180'


# 1. Read obj1 and obj2 as Tensors
obj1 = Gaussian().readGaussian(obj1_path + '/point_cloud.ply')
obj2 = Gaussian().readGaussian(obj2_path + '/point_cloud.ply')
obj1.centroids, obj1.rots = apply_transform_torch(obj1.centroids, obj1.rots,
                                                   torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=device),
                                                   torch.tensor([0.0, 0.0, 0.0], device=device))
save_points_as_ply(obj1.centroids.cpu().numpy(), obj1_path + '/point_cloud_input.ply')
    

# 2. (Optional) Align centers of mass in-place
with torch.no_grad():
    obj1.centroids -= obj1.centroids.mean(dim=0, keepdim=True)
    obj2.centroids -= obj2.centroids.mean(dim=0, keepdim=True)

# 3. Construct a naive feature as [opacity, scales, rots]
obj1.feature = torch.cat([obj1.opacity, obj1.scales, obj1.rots], dim=1)
obj2.feature = torch.cat([obj2.opacity, obj2.scales, obj2.rots], dim=1)

# 4. Sample random quaternions in PyTorch
sample_cnt = 100
sampled_quaternions = sample_uniform_quaternions_torch(sample_cnt, device=device)  # [100, 4]

opt_iter = 100
q_list = []
t_list = []
loss_list = []

# We'll create a "base copy" of obj1 data that we do NOT alter,
# so we can re-transform it each time in a consistent way.
base_centroids = obj1.centroids.clone()
base_rots      = obj1.rots.clone()
base_features  = obj1.feature.clone()

for i in tqdm(range(sample_cnt), desc="Quaternion Samples"):
    # The variable q, t we will optimize
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device, requires_grad=True)
    t = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)

    # Apply the random "pre-rotation" to obj1 data (just once at the start of this sample)
    # so we have "obj1_cup" that starts from a random orientation
    random_q = sampled_quaternions[i]
    new_centroids, new_rots = apply_transform_torch(base_centroids, base_rots, random_q, torch.zeros(3, device=device))

    # Build a temporary "Gaussian" for the transformed obj1
    obj1_temp = Gaussian()
    obj1_temp.centroids = new_centroids
    obj1_temp.rots      = new_rots
    obj1_temp.opacity   = obj1.opacity  # unchanged
    obj1_temp.scales    = obj1.scales   # unchanged
    # Recompute feature for the temporary
    obj1_temp.feature   = torch.cat([obj1_temp.opacity, obj1_temp.scales, obj1_temp.rots], dim=1)

    optimizer = optim.Adam([q, t], lr=1e-2)

    for step in range(opt_iter):
        optimizer.zero_grad()
        loss_val = loss_fn_torch(obj1_temp, obj2, q, t)
        loss_val.backward()
        optimizer.step()

    q_list.append(q.detach().cpu().numpy())
    t_list.append(t.detach().cpu().numpy())
    loss_list.append(loss_val.item())

# 5. Find minimum loss index and compute final transform
min_loss_idx = np.argmin(loss_list)
best_q_torch = torch.from_numpy(q_list[min_loss_idx]).float().to(device)
random_q_torch = sampled_quaternions[min_loss_idx].to(device)

# Combine the random quaternion (which was applied at the start)
# We'll just do it in torch:
best_q_torch = quaternion_multiply_torch(best_q_torch.unsqueeze(0),
                                         random_q_torch.unsqueeze(0)).squeeze(0)
best_q_torch = best_q_torch / (best_q_torch.norm() + 1e-9)

best_t_torch = torch.from_numpy(t_list[min_loss_idx]).float().to(device)

# 6. Apply final transform to the original obj1
final_centroids, final_rots = apply_transform_torch(base_centroids, base_rots, best_q_torch, best_t_torch)

# If you want a final "Gaussian" object containing the final result:
final_obj_1 = Gaussian()
final_obj_1.centroids = final_centroids
final_obj_1.rots      = final_rots
final_obj_1.feature   = torch.cat([obj1.opacity, obj1.scales, final_obj_1.rots], dim=1)

# 7. Save or export `final_cup_1.centroids` back to CPU / NumPy for writing .ply, etc.
final_points_np = final_obj_1.centroids.detach().cpu().numpy().astype(np.float32)
print("Transformed centroids shape:", final_points_np.shape)

save_points_as_ply(final_points_np, obj1_path + '/point_cloud_transformed.ply')
print(f"Saved transformed points to {obj1_path + '/point_cloud_transformed.ply'}")

# 8. Save all quarternions and translations to a fil
transform_output_path = obj1_path + '/transform_results.txt'
with open(transform_output_path, 'w') as f:
    f.write(f"Best quaternion: {best_q_torch.cpu().numpy()}\n")
    f.write(f"Best translation: {best_t_torch.cpu().numpy()}\n")
    for i in range(sample_cnt):
        # Combine optimized quaternion q_list[i] * random sampled_quaternions[i]
        cur_q = torch.from_numpy(q_list[i]).float().to(device)
        cur_random_q = sampled_quaternions[i]
        combined_q = quaternion_multiply_torch(cur_q.unsqueeze(0), cur_random_q.unsqueeze(0)).squeeze(0)
        combined_q = combined_q / (combined_q.norm() + 1e-9)

        # Write the final combined quaternion and the translation
        combined_q_np = combined_q.detach().cpu().numpy()
        f.write(f"Sample {i}: combined_q={combined_q_np}, t={t_list[i]}\n")

print(f"Saved quaternion and translation results to {transform_output_path}")



