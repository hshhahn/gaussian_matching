#!/usr/bin/env python3
"""
Align two point clouds by optimising an SE(3) transform that minimises a
bidirectional nearest-neighbour loss.

Example
-------
python match_random_adam.py \
    --src demo_0_red_points.ply \
    --tgt demo_1_red_points.ply \
    --out_ply aligned.ply \
    --out_txt transform.txt
"""
import argparse
import numpy as np
import torch
import torch.optim as optim
from plyfile import PlyData, PlyElement
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#                               I/O utilities                                 #
# --------------------------------------------------------------------------- #
def load_ply_points(path):
    """Load only XYZ from a binary/text PLY → (N,3) float32 torch tensor."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return torch.from_numpy(pts).to(device)


def save_ply_points(pts, path):
    """Save an (N,3) tensor (on *CPU*) to a binary PLY at *path*."""
    pts = pts.cpu().numpy().astype(np.float32)
    verts = np.zeros(pts.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=False).write(path)


# --------------------------------------------------------------------------- #
#                         Quaternion & transform helpers                      #
# --------------------------------------------------------------------------- #
def quaternion_to_rotation_matrix(q):
    q = q / (q.norm() + 1e-9)
    w, x, y, z = q
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )


def sample_uniform_quaternions(n):
    """(n,4) tensor of (w,x,y,z) quaternions uniformly sampled on S³."""
    u1, u2, u3 = torch.rand(n, device=device), torch.rand(n, device=device), torch.rand(n, device=device)
    q0 = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
    q1 = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
    q2 = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
    q3 = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)
    return torch.stack([q3, q0, q1, q2], dim=1)  # (w,x,y,z)


def quaternion_multiply(q1, q2):
    """Hamilton product of two (4,) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        device=q1.device,
    )


def transform_points(pts, q, t):
    """Apply rotation *q* and translation *t* to (N,3) points."""
    R = quaternion_to_rotation_matrix(q)  # (3,3)
    return pts @ R.T + t


# --------------------------------------------------------------------------- #
#                                 Losses                                      #
# --------------------------------------------------------------------------- #
def loss_fn_bidirectional_fast(src_pts, tgt_pts, q, t, chunk_size=512):
    """
    Bidirectional nearest-neighbour loss (src→tgt and tgt→src) evaluated in
    chunks to keep memory under control.
    """
    src2 = transform_points(src_pts, q, t)
    loss = torch.tensor(0.0, device=src2.device)

    # src → tgt
    for i in range(0, src2.size(0), chunk_size):
        chunk = src2[i : i + chunk_size]
        d2 = torch.cdist(chunk, tgt_pts).pow(2)
        loss = loss + d2.min(dim=1).values.sum()

    # tgt → src
    for j in range(0, tgt_pts.size(0), chunk_size):
        chunk = tgt_pts[j : j + chunk_size]
        d2 = torch.cdist(chunk, src2).pow(2)
        loss = loss + d2.min(dim=1).values.sum()

    return loss


# --------------------------------------------------------------------------- #
#                                  Main                                       #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Rigid alignment of two point clouds.")
    parser.add_argument("--src", "-s", required=True, help="Path to SOURCE PLY")
    parser.add_argument("--tgt", "-t", required=True, help="Path to TARGET PLY")
    parser.add_argument("--out_ply", default="source_aligned_world.ply", help="Output aligned PLY")
    parser.add_argument("--out_txt", default="transform_world.txt", help="Output transform text file")
    parser.add_argument("--seeds", "-M", type=int, default=100, help="# random quaternion seeds")
    parser.add_argument("--iters", type=int, default=100, help="# optimisation steps per seed")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    args = parser.parse_args()

    # ------------------------------------------------------------------ load
    src_orig = load_ply_points(args.src)
    tgt_orig = load_ply_points(args.tgt)

    # centre both clouds
    src_center, tgt_center = src_orig.mean(0, keepdim=True), tgt_orig.mean(0, keepdim=True)
    src, tgt = src_orig - src_center, tgt_orig - tgt_center

    # ---------------------------------------------------------- multi-start
    pre_q = sample_uniform_quaternions(args.seeds)
    best_loss, best_q, best_t = float("inf"), None, None

    for i in tqdm(range(args.seeds)):
        src_seed = transform_points(src, pre_q[i], torch.zeros(3, device=device))
        q = torch.tensor([1.0, 0, 0, 0], device=device, requires_grad=True)
        t = torch.zeros(3, device=device, requires_grad=True)
        opt = optim.Adam([q, t], lr=args.lr)

        for _ in range(args.iters):
            opt.zero_grad(set_to_none=True)
            l = loss_fn_bidirectional_fast(src_seed, tgt, q, t)
            l.backward()
            opt.step()

        if l.item() < best_loss:
            best_loss = l.item()
            best_q = quaternion_multiply(q.detach(), pre_q[i])
            best_q = best_q / (best_q.norm() + 1e-9)
            best_t = t.detach()

    # ----------------------------------------------------------- compose
    print(f"Best loss: {best_loss:.4f}")
    R = quaternion_to_rotation_matrix(best_q)
    full_t = -R @ src_center.squeeze(0) + best_t + tgt_center.squeeze(0)

    final_src = (src_orig @ R.T) + full_t
    save_ply_points(final_src, args.out_ply)

    with open(args.out_txt, "w") as f:
        f.write(f"rotation (quaternion w,x,y,z): {best_q.cpu().numpy()}\n")
        f.write(f"translation (x,y,z):          {full_t.cpu().numpy()}\n")

    print(f"Aligned cloud  → {args.out_ply}")
    print(f"Transform saved → {args.out_txt}")


if __name__ == "__main__":
    main()
