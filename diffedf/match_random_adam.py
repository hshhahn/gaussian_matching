import numpy as np
import torch
import torch.optim as optim
from plyfile import PlyData, PlyElement
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ply_points(path):
    """Load only XYZ from a binary/text PLY into an (N,3) float32 torch tensor."""
    ply = PlyData.read(path)
    v = ply['vertex']
    pts = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    return torch.from_numpy(pts).to(device)


def save_ply_points(pts, path):
    """Save an (N,3) torch tensor (on CPU) to a binary PLY."""
    pts = pts.cpu().numpy().astype(np.float32)
    verts = np.zeros(pts.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4')])
    verts['x'], verts['y'], verts['z'] = pts[:,0], pts[:,1], pts[:,2]
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(path)


def quaternion_to_rotation_matrix(q):
    q = q / (q.norm() + 1e-9)
    w, x, y, z = q          # keep these as tensors
    R = torch.stack([       # 3 × 3 with gradients
        torch.stack([1-2*(y*y+z*z), 2*(x*y - w*z),   2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1-2*(x*x+z*z),   2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x),   1-2*(x*x+y*y)])
    ])
    return R


def sample_uniform_quaternions(n):
    """Returns (n,4) tensor of (w,x,y,z) uniformly sampled."""
    u1 = torch.rand(n, device=device)
    u2 = torch.rand(n, device=device)
    u3 = torch.rand(n, device=device)
    q0 = torch.sqrt(1-u1)*torch.sin(2*np.pi*u2)
    q1 = torch.sqrt(1-u1)*torch.cos(2*np.pi*u2)
    q2 = torch.sqrt(u1)*torch.sin(2*np.pi*u3)
    q3 = torch.sqrt(u1)*torch.cos(2*np.pi*u3)
    return torch.stack([q3, q0, q1, q2], dim=1)  # note ordering (w,x,y,z)


def quaternion_multiply(q1, q2):
    """Elementwise multiply two (4,) quaternions."""
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], device=q1.device)


def transform_points(pts, q, t):
    """Apply rotation q and translation t to (N,3) pts."""
    R = quaternion_to_rotation_matrix(q)   # (3,3)
    return pts @ R.T + t


def loss_fn_bidirectional(src_pts, tgt_pts, q, t):
    """
    Bidirectional nearest‐neighbor loss:
      sum over src_pts → closest tgt_pts
    plus
      sum over tgt_pts → closest src_pts (after transform).
    """
    # 1) Transform your source points
    src2 = transform_points(src_pts, q, t)   # (Ns,3)

    loss = torch.tensor(0.0, device=src2.device)

    # 2) src → tgt
    for p in src2:  
        diff2 = (p.unsqueeze(0) - tgt_pts).pow(2).sum(dim=1)  
        loss = loss + torch.min(diff2)

    # 3) tgt → src
    for p in tgt_pts:
        diff2 = (p.unsqueeze(0) - src2).pow(2).sum(dim=1)
        loss = loss + torch.min(diff2)

    return loss

def loss_fn(src_pts, tgt_pts, q, t):
    """
    Sum of squared nearest‐neighbor distances from src→tgt
    without building the full NxM matrix.
    """
    # 1) Transform your source points
    src2 = transform_points(src_pts, q, t)   # (Ns,3)

    # 2) For each transformed source point, compute squared dists to all targets
    loss = torch.tensor(0.0, device=src2.device)
    for p in src2:  
        # p: (3,), tgt_pts: (Nt,3) → diff: (Nt,3)
        diff2 = (p.unsqueeze(0) - tgt_pts).pow(2).sum(dim=1)  
        # take the closest
        loss = loss + torch.min(diff2)

    return loss

def loss_fn_bidirectional_fast(src_pts, tgt_pts, q, t, chunk_size=512):
    """
    Bidirectional nearest‐neighbor loss, chunked:
      sum_{p∈src} min_{q∈tgt} ||p-q||²
    + sum_{q∈tgt} min_{p∈src_trans} ||p-q||²
    """
    src2 = transform_points(src_pts, q, t)  # (Ns,3)
    loss = torch.tensor(0.0, device=src2.device)

    # src → tgt
    for i in range(0, src2.size(0), chunk_size):
        chunk = src2[i:i+chunk_size]                     # (C,3)
        d2 = torch.cdist(chunk, tgt_pts, p=2.0).pow(2)   # (C, Nt)
        loss = loss + d2.min(dim=1).values.sum()

    # tgt → src
    for j in range(0, tgt_pts.size(0), chunk_size):
        chunk = tgt_pts[j:j+chunk_size]                  # (C,3)
        d2 = torch.cdist(chunk, src2, p=2.0).pow(2)      # (C, Ns)
        loss = loss + d2.min(dim=1).values.sum()

    return loss


if __name__ == "__main__":
    # paths
    src_path = "demo_0_red_points.ply"
    tgt_path = "demo_1_red_points.ply"

    # load
    src_orig = load_ply_points(src_path)  # (Ns,3)
    tgt_orig = load_ply_points(tgt_path)  # (Nt,3)
    
    src_center = src_orig.mean(dim=0, keepdim=True)
    tgt_center = tgt_orig.mean(dim=0, keepdim=True)

    src = src_orig - src_center
    tgt = tgt_orig - tgt_center 

    # sample pre‐rotations
    M = 100
    pre_q = sample_uniform_quaternions(M)

    best_loss = float('inf')
    best_q   = None
    best_t   = None

    # loop over seeds
    for i in tqdm(range(M)):
        # apply the pre‐rotation once
        src_seed = transform_points(src, pre_q[i], torch.zeros(3,device=device))

        # optimize a fresh q (init identity) and t=0
        q = torch.tensor([1.0,0,0,0], device=device, requires_grad=True)
        t = torch.zeros(3, device=device, requires_grad=True)

        opt = optim.Adam([q,t], lr=1e-2)
        for _ in range(100):
            opt.zero_grad()
            l = loss_fn_bidirectional_fast(src_seed, tgt, q, t)
            # l = loss_fn_bidirectional(src_seed, tgt, q, t)
            # l = loss_fn(src_seed, tgt, q, t)  
            l.backward()
            opt.step()

        if l.item() < best_loss:
            best_loss = l.item()
            # combine pre‐rotation and learned rotation
            combined_q = quaternion_multiply(q.detach(), pre_q[i])
            best_q = combined_q / (combined_q.norm() + 1e-9)
            best_t = t.detach()

    print(f"Best loss: {best_loss:.4f}")
    src_mean = src_center.squeeze(0)   # (3,) tensor
    tgt_mean = tgt_center.squeeze(0)

    R = quaternion_to_rotation_matrix(best_q)
    full_t = -R @ src_mean + best_t + tgt_mean    # equation above

    # transform the ORIGINAL (un-centred) source cloud
    final_src = (src_orig @ R.T) + full_t         # (Ns,3)

    save_ply_points(final_src, "source_aligned_world.ply")

    with open("transform_world.txt","w") as f:
        f.write(f"rotation (quaternion w,x,y,z): {best_q.cpu().numpy()}\n")
        f.write(f"translation (x,y,z):          {full_t.cpu().numpy()}\n")

    print("Saved aligned pointcloud → source_aligned.ply")
    print("Transform saved → transform.txt")
