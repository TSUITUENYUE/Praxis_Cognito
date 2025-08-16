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
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)
    aa = axis_angle.to(torch.float32)
    B = aa.shape[0]

    theta = aa.norm(dim=1)
    eps = 1e-8
    theta_safe = torch.clamp(theta, min=eps)
    theta2 = theta * theta

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    A_exact = sin_t / theta_safe
    B_exact = (1. - cos_t) / torch.clamp(theta2, min=eps)

    th2 = theta2
    th4 = th2 * th2
    A_series = 1. - th2/6. + th4/120.
    B_series = 0.5 - th2/24. + th4/720.

    small = (theta < 1e-3)
    A  = torch.where(small, A_series, A_exact).view(B, 1, 1)
    Bc = torch.where(small, B_series, B_exact).view(B, 1, 1)

    K = _skew(aa)  # [B,3,3]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(B, 3, 3)
    return I + A * K + Bc * (K @ K)



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
        order = self.computation_order

        # Precompute Python lists (no tensors → no .item() later)
        self.order_parent_idx_py = [self.link_names.index(p) for (p, _, _) in order]
        self.order_child_idx_py = [self.link_names.index(c) for (_, c, _) in order]

        self.order_is_rev_py = []
        self.order_dof_idx_py = []
        self.order_axes = []  # keep as tensor per-edge, but we’ll index by python int
        self.order_Torigin = []  # keep as tensor per-edge, but we’ll index by python int
        for (_, _, jn) in order:
            jd = self.joint_data[jn]
            is_rev = (jd["type"] == "revolute")
            self.order_is_rev_py.append(is_rev)
            self.order_dof_idx_py.append(self.joint_idx.get(jn, -1) if is_rev else -1)
            self.order_axes.append(jd.get("axis_tensor", torch.zeros(3, device=self.device)))
            self.order_Torigin.append(jd["T_origin"])

        # stack the per-edge constants once (tensors are fine to index by python ints)
        self.order_axes = torch.stack(self.order_axes, dim=0).to(torch.float32)  # [E,3]
        self.order_Torigin = torch.stack(self.order_Torigin, dim=0).to(torch.float32)  # [E,4,4]

        self.E = len(order)
        self.num_links = len(self.link_names)
        # positions are emitted in link_names order; caching avoids per-call .index
        self._pos_link_indices_py = list(range(self.num_links))
        self.register_buffer("order_axes_buf", self.order_axes)
        self.register_buffer("order_Torigin_buf", self.order_Torigin)

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
        with autocast(False):  # keep numerics identical
            if joint_angles.dim() == 1:
                joint_angles = joint_angles.unsqueeze(0)

            dev = self.eye_4.device  # stick to module's device for safety
            q = joint_angles.to(dtype=torch.float32, device=dev)  # [B, DoF]
            B = q.shape[0]

            # --- allocate all link transforms at once: [B, L, 4, 4], fill with identity
            link_T_all = self.eye_4.view(1, 1, 4, 4).expand(B, self.num_links, 4, 4).clone()

            # scratch buffer for a joint transform (reused every edge)
            T_joint_buf = self.eye_4.view(1, 4, 4).expand(B, 4, 4).clone()

            # choose constant source (buffer if registered; else tensors)
            order_axes = getattr(self, "order_axes_buf", self.order_axes)  # [E,3]
            order_Torig = getattr(self, "order_Torigin_buf", self.order_Torigin)  # [E,4,4]

            for e in range(self.E):
                p_idx = self.order_parent_idx_py[e]  # python int
                c_idx = self.order_child_idx_py[e]

                # parent world transform
                T_p = link_T_all[:, p_idx, :, :]  # [B,4,4]

                # constant origin transform for this joint
                T_origin_b = order_Torig[e].unsqueeze(0).expand(B, 4, 4)  # [B,4,4]

                if self.order_is_rev_py[e]:
                    # joint rotation (Rodrigues) and full 4x4 transform
                    dof_idx = self.order_dof_idx_py[e]  # python int
                    angle = q[:, dof_idx]  # [B]
                    axis = order_axes[e].unsqueeze(0).expand(B, 3)  # [B,3]
                    Rj = rotmat_axis_angle_stable(axis * angle.unsqueeze(1))  # [B,3,3]

                    # reset scratch to identity and insert rotation in-place
                    T_joint_buf.copy_(self.eye_4)  # [B,4,4] all identity
                    T_joint_buf[:, :3, :3] = Rj

                    # T_child = T_p @ T_origin @ T_joint
                    T_pc = torch.bmm(T_p, T_origin_b)  # [B,4,4]
                    T_child = torch.bmm(T_pc, T_joint_buf)
                else:
                    # fixed/prismatic treated as identity joint here => only origin transform
                    T_child = torch.bmm(T_p, T_origin_b)  # [B,4,4]

                # write back
                link_T_all[:, c_idx, :, :] = T_child

            # positions in link_names order
            positions = link_T_all[:, :, :3, 3].reshape(B, 3 * self.num_links)  # [B, 3*links]
            return positions

