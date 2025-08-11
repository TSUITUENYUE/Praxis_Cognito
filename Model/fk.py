import torch
import torch.nn as nn
from xml.etree import ElementTree as ET
from collections import defaultdict
from torch.cuda.amp import autocast

# --- utilities ---

def vectorized_euler_to_rot_matrix(rpy: torch.Tensor):
    """
    rpy: [...,3] or [3]. Returns rotation matrix 3x3 (if bs==1) or [B,3,3].
    Used only for URDF origins (constants), so the original behavior is kept.
    """
    if rpy.dim() == 1:
        rpy = rpy.unsqueeze(0)
    bs = rpy.shape[0]
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)

    R = torch.zeros(bs, 3, 3, device=rpy.device, dtype=rpy.dtype)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    if bs == 1:
        return R.squeeze(0)
    return R

def _skew(v: torch.Tensor):
    """v: [B,3] -> [B,3,3] skew-symmetric matrices."""
    B = v.shape[0]
    K = torch.zeros(B, 3, 3, device=v.device, dtype=v.dtype)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    K[:, 0, 1] = -vz; K[:, 0, 2] =  vy
    K[:, 1, 0] =  vz; K[:, 1, 2] = -vx
    K[:, 2, 0] = -vy; K[:, 2, 1] =  vx
    return K

def rotmat_axis_angle_stable(axis_angle: torch.Tensor):
    """
    axis_angle: [B,3] (axis * angle, axis need not be unit).
    Returns [B,3,3]. Smooth at theta -> 0, with Taylor expansions for A,B.
    """
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)
    B = axis_angle.shape[0]
    aa = axis_angle
    theta = aa.norm(dim=1)                       # [B]
    eps = 1e-8
    theta2 = theta * theta + eps
    A = torch.sin(theta) / (theta + eps)         # sinθ/θ
    Bc = (1.0 - torch.cos(theta)) / theta2       # (1-cosθ)/θ^2

    # Taylor for very small angles
    small = theta < 1e-3
    if small.any():
        th = theta[small]
        A_t = 1.0 - (th**2)/6.0 + (th**4)/120.0
        B_t = 0.5 - (th**2)/24.0 + (th**4)/720.0
        A = A.clone(); Bc = Bc.clone()
        A[small]  = A_t
        Bc[small] = B_t

    K = _skew(aa)                                # [B,3,3]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(B, 3, 3)
    A = A.view(B, 1, 1); Bc = Bc.view(B, 1, 1)
    R = I + A * K + Bc * (K @ K)                 # Rodrigues
    return R

# --- FK model ---

class FKModel(nn.Module):
    def __init__(self, urdf):
        super(FKModel, self).__init__()
        self.parse_urdf_file(urdf)
        # default device for buffers/constants
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Precompute traversal order (parent -> child)
        self.computation_order = []
        stack = ['base']
        visited = set()
        while stack:
            link = stack.pop()
            if link in visited:
                continue
            visited.add(link)
            for child, joint_name in self.child_map.get(link, []):
                self.computation_order.append((link, child, joint_name))
                stack.append(child)

        # Buffers
        self.register_buffer('eye_4', torch.eye(4, device=self.device, dtype=torch.float32))
        self.joint_idx = {name: i for i, name in enumerate(self.joint_names)}

        # Precompute constant transforms per joint
        self.joint_data = {}
        for joint_name, joint in self.joint_map.items():
            # origin transform (URDF constant)
            xyz = torch.tensor(joint['xyz'], device=self.device, dtype=torch.float32)
            rpy = torch.tensor(joint['rpy'], device=self.device, dtype=torch.float32)
            R = vectorized_euler_to_rot_matrix(rpy)  # [3,3]
            T_origin = self.eye_4.clone()
            T_origin[:3, :3] = R
            T_origin[:3, 3]  = xyz
            data = {'type': joint['type'], 'parent': joint['parent'], 'child': joint['child'],
                    'T_origin': T_origin}
            if joint['type'] == 'revolute':
                axis = torch.tensor(joint['axis'], device=self.device, dtype=torch.float32)
                # keep axis unnormalized; axis-angle uses axis * angle
                data['axis_tensor'] = axis
            else:
                data['T_joint'] = self.eye_4.clone()
            self.joint_data[joint_name] = data

    def parse_urdf_file(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        self._parse_urdf_root(root)

    def _parse_urdf_root(self, root):
        self.link_names = [link.get('name') for link in root.findall('link')]
        # only revolute joints are actuated; but keep all joints in maps
        self.joint_names = [j.get('name') for j in root.findall('joint') if j.get('type') == 'revolute']
        self.joint_map = {}
        self.child_map = defaultdict(list)
        self.parent_map = {}
        for joint in root.findall('joint'):
            name = joint.get('name')
            jtype = joint.get('type')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            self.child_map[parent].append((child, name))
            self.parent_map[child] = parent
            origin = joint.find('origin')
            xyz = list(map(float, (origin.get('xyz') or "0 0 0").split())) if origin is not None else [0.0, 0.0, 0.0]
            rpy = list(map(float, (origin.get('rpy') or "0 0 0").split())) if origin is not None else [0.0, 0.0, 0.0]
            axis_elem = joint.find('axis')
            axis = list(map(float, (axis_elem.get('xyz') or "1 0 0").split())) if axis_elem else [1.0, 0.0, 0.0]
            self.joint_map[name] = {'type': jtype, 'parent': parent, 'child': child,
                                    'xyz': xyz, 'rpy': rpy, 'axis': axis}

    def forward(self, joint_angles: torch.Tensor):
        """
        joint_angles: [B, DoF] or [DoF]
        returns: [B, len(link_names)*3] (never squeezed; always batched)
        """
        # Force FK in fp32 for stable trig/gradients (even under AMP)
        with autocast(False):
            if joint_angles.dim() == 1:
                joint_angles = joint_angles.unsqueeze(0)
            joint_angles = joint_angles.to(dtype=torch.float32)

            B = joint_angles.shape[0]
            # base transform
            transforms = {'base': self.eye_4.unsqueeze(0).expand(B, 4, 4).clone()}

            for parent, child, joint_name in self.computation_order:
                jd = self.joint_data[joint_name]
                T_origin = jd['T_origin'].unsqueeze(0).expand(B, 4, 4)  # [B,4,4]

                if jd['type'] == 'revolute':
                    idx = self.joint_idx.get(joint_name, None)
                    if idx is None:
                        # Non-actuated revolute (unlikely): treat as fixed identity
                        T_joint = self.eye_4.unsqueeze(0).expand(B, 4, 4)
                    else:
                        angle = joint_angles[:, idx]                         # [B]
                        axis  = jd['axis_tensor'].unsqueeze(0).expand(B, 3)  # [B,3]
                        aa    = axis * angle.unsqueeze(1)                    # [B,3]
                        Rj    = rotmat_axis_angle_stable(aa)                 # [B,3,3]
                        T_joint = self.eye_4.unsqueeze(0).expand(B, 4, 4).clone()
                        T_joint[:, :3, :3] = Rj
                else:
                    T_joint = jd['T_joint'].unsqueeze(0).expand(B, 4, 4)

                T_parent = transforms[parent]                               # [B,4,4]
                # T_child = T_parent @ T_origin @ T_joint
                T_child = T_parent.matmul(T_origin).matmul(T_joint)
                transforms[child] = T_child

            # collect link positions in world
            pos_list = []
            for link in self.link_names:
                T = transforms.get(link, self.eye_4.unsqueeze(0).expand(B, 4, 4))
                pos_list.append(T[:, :3, 3])                                # [B,3]
            positions = torch.cat(pos_list, dim=1)                           # [B, 3*links]
            return positions
