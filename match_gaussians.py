from plyfile import PlyData
import numpy as np

class Gaussian:
    def __init__(self):
        self.centroids = None
        self.normals = None
        self.opacity = None
        self.scales = None
        self.rots = None
        self.sh_base = None

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

# Example usage
cup1 = Gaussian()
cup1.readGaussian('cup/test/cup_0082/point_cloud.ply')

cup2 = Gaussian()
cup2.readGaussian('cup/test/cup_0096/point_cloud.ply')

# Rotate the `rots` attribute of cup2 using an input quaternion
input_rotation = np.array([0.707, 0.707, 0, 0])  # Example input quaternion (90 degrees around X-axis)
input_rotation = normalize_quaternion(input_rotation)  # Normalize the input quaternion

# Convert the input quaternion to a rotation matrix
rotation_matrix = quaternion_to_rotation_matrix(input_rotation)

# Rotate centroids of cup2
cup2.centroids = np.dot(cup2.centroids, rotation_matrix.T)

# Apply the input rotation to all quaternions in cup2.rots
for i in range(cup2.rots.shape[0]):
    cup2.rots[i] = quaternion_multiply(input_rotation, cup2.rots[i])
    cup2.rots[i] = normalize_quaternion(cup2.rots[i])  # Normalize the result
