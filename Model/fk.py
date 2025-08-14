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
    # v: [N,3] -> [N,3,3]
    N = v.shape[0]
    K = v.new_zeros(N, 3, 3)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    K[:, 0, 1] = -z; K[:, 0, 2] =  y
    K[:, 1, 0] =  z; K[:, 1, 2] = -x
    K[:, 2, 0] = -y; K[:, 2, 1] =  x
    return K

def rotmat_axis_angle_stable(axis_angle: torch.Tensor):
    """
    axis_angle: [N,3] or [3]; returns [N,3,3].
    Uses series expansion near 0 with torch.where (no masked assignment).
    """
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)
    aa = axis_angle.to(torch.float32)
    N = aa.shape[0]

    theta = aa.norm(dim=-1, keepdim=True)                # [N,1]
    eps = 1e-8
    theta_safe = torch.clamp(theta, min=eps)
    theta2 = theta * theta

    # exact
    A_exact = torch.sin(theta) / theta_safe              # sinθ/θ
    B_exact = (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=eps)  # (1-cos)/θ^2

    # series near 0
    th2 = theta2
    th4 = th2 * th2
    A_series = 1.0 - th2/6.0 + th4/120.0
    B_series = 0.5 - th2/24.0 + th4/720.0

    small = (theta < 1e-3)
    A  = torch.where(small, A_series, A_exact)           # [N,1]
    Bc = torch.where(small, B_series, B_exact)           # [N,1]

    K = _skew(aa)                                        # [N,3,3]
    I = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(N, 3, 3)
    return I + A[..., None, None]*K + Bc[..., None, None]*(K @ K)


# ---------- tensorized FK (no dicts on the hot path) ----------

class FKModel(nn.Module):
    """
    TorchDynamo / torch.compile friendly FK:
      - URDF parsed once → tensors (buffers)
      - Forward takes joint angles [B, DoF] (revolute/continuous only)
      - Returns link world positions flattened: [B, 3 * num_links]
    """
    def __init__(self, urdf_path: str):
        super().__init__()

        # Parse URDF → graph
        (link_names,                   # [L] list[str]
         parent_names, child_names,    # [E] list[str]
         joint_types,                  # [E] list[str]
         origins_xyz, origins_rpy,     # [E,3], [E,3]
         axes,                         # [E,3]
         revolute_joint_names) = self._parse_urdf(urdf_path)

        self.link_names = link_names  # keep for reference

        # Map links to indices
        link_to_idx = {name: i for i, name in enumerate(link_names)}
        E = len(parent_names)
        L = len(link_names)

        # Edge tensors
        parent_idx = torch.tensor([link_to_idx[p] for p in parent_names], dtype=torch.long)
        child_idx  = torch.tensor([link_to_idx[c] for c in child_names], dtype=torch.long)

        # Joint type ids: 0=fixed, 1=revolute, 2=prismatic (we treat prismatic as fixed by default)
        type_id = []
        for jt in joint_types:
            if jt in ("revolute", "continuous"):
                type_id.append(1)
            elif jt == "prismatic":
                type_id.append(2)
            else:
                type_id.append(0)
        type_id = torch.tensor(type_id, dtype=torch.long)

        # Constant origin transforms per edge (4x4)
        R0 = self._euler_to_rotmat(torch.tensor(origins_rpy, dtype=torch.float32))   # [E,3,3]
        T_origin = torch.eye(4, dtype=torch.float32).repeat(E, 1, 1)
        T_origin[:, :3, :3] = R0
        T_origin[:, :3,  3] = torch.tensor(origins_xyz, dtype=torch.float32)

        # Revolute DoF mapping: order of actuated joints = order in URDF for revolute/continuous
        # map edge→dof_idx (−1 for non-actuated)
        dof_map = torch.full((E,), -1, dtype=torch.long)
        name_to_dof = {n: i for i, n in enumerate(revolute_joint_names)}
        # We need edge names to fill; re-run parse to get joint names aligned with edges
        edge_joint_names = self._parse_joint_names(urdf_path)
        for e, jn in enumerate(edge_joint_names):
            if jn in name_to_dof and type_id[e] == 1:
                dof_map[e] = name_to_dof[jn]

        # Axes
        axes_t = torch.tensor(axes, dtype=torch.float32)  # [E,3]

        # Compute a topological (parent→child) order for edges (fixed, deterministic)
        edge_order = self._topo_edge_order(L, parent_idx, child_idx)

        # Register buffers (constant during training; device will follow .to())
        self.register_buffer("parent_idx", parent_idx, persistent=False)
        self.register_buffer("child_idx",  child_idx,  persistent=False)
        self.register_buffer("type_id",    type_id,    persistent=False)
        self.register_buffer("T_origin",   T_origin,   persistent=False)  # [E,4,4]
        self.register_buffer("axes",       axes_t,     persistent=False)  # [E,3]
        self.register_buffer("dof_map",    dof_map,    persistent=False)  # [E]
        self.register_buffer("edge_order", edge_order, persistent=False)  # [E]
        self.L = L
        self.E = E
        self.D = len(revolute_joint_names)  # expected DoF

        # handy identity
        self.register_buffer("eye4", torch.eye(4, dtype=torch.float32), persistent=False)

    # ---------------- public API ----------------

    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        joint_angles: [B, DoF] OR [DoF]
        returns world positions of all links flattened: [B, 3*L]
        """
        with autocast(False):  # keep FK in fp32 for stable trig/grad
            if joint_angles.dim() == 1:
                joint_angles = joint_angles.unsqueeze(0)
            q = joint_angles.to(dtype=torch.float32)               # [B, D]
            B = q.shape[0]
            assert q.shape[1] == self.D, f"Expected DoF={self.D}, got {q.shape[1]}"

            # Gather per-edge actuation value (angle or disp). Non-actuated → 0.
            # map indices <0 to 0 then zero them with mask.
            safe_idx = torch.clamp(self.dof_map, min=0)            # [E]
            vals = q[:, safe_idx]                                  # [B, E]
            actuated = (self.dof_map >= 0)                         # [E]
            vals = vals * actuated.to(vals.dtype)                  # zero for non-actuated edges

            # Build T_joint per edge, per batch (vectorized over edges)
            # Revolute: R(axis*angle), t=0
            axis_angle = self.axes[None, :, :] * vals[:, :, None]  # [B,E,3]
            R = rotmat_axis_angle_stable(axis_angle.reshape(-1, 3)).reshape(B, self.E, 3, 3)  # [B,E,3,3]
            T_joint = self.eye4.repeat(B, self.E, 1, 1)            # [B,E,4,4]
            # Set rotation only for revolute edges
            rev_mask = (self.type_id == 1)[None, :, None, None]    # [1,E,1,1]
            T_joint[:, :, :3, :3] = torch.where(rev_mask, R, T_joint[:, :, :3, :3])

            # (Optional) Prismatic support: uncomment to enable
            # pris_mask = (self.type_id == 2)[None, :, None]
            # trans = self.axes[None, :, :] * vals[:, :, None]      # [B,E,3]
            # T_joint[:, :, :3, 3] = torch.where(pris_mask.expand(B, self.E, 3),
            #                                    trans, T_joint[:, :, :3, 3])

            # Compose with constant origin: T_local = T_origin @ T_joint
            T_origin = self.T_origin[None, :, :, :].expand(B, self.E, 4, 4)
            T_local  = T_origin @ T_joint                           # [B,E,4,4]

            # Accumulate along the tree in a fixed topological edge order
            T_links = self.eye4.expand(B, 4, 4).unsqueeze(1).repeat(1, self.L, 1, 1).clone()  # [B,L,4,4]
            for e in self.edge_order.tolist():  # integer loop, compiler-friendly
                p = int(self.parent_idx[e].item())
                c = int(self.child_idx[e].item())
                # T_child = T_parent @ T_local[e]
                T_links[:, c] = T_links[:, p] @ T_local[:, e]

            # Collect world positions for all links
            pos = T_links[:, :, :3, 3].reshape(B, 3 * self.L)       # [B, 3*L]
            return pos

    # ---------------- internals ----------------

    @staticmethod
    def _euler_to_rotmat(rpy: torch.Tensor) -> torch.Tensor:
        # rpy: [E,3] → [E,3,3]
        r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        cr, sr = torch.cos(r), torch.sin(r)
        cp, sp = torch.cos(p), torch.sin(p)
        cy, sy = torch.cos(y), torch.sin(y)
        E = r.shape[0]
        R = rpy.new_zeros(E, 3, 3)
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        return R

    @staticmethod
    def _parse_urdf(urdf_path):
        root = ET.parse(urdf_path).getroot()

        # Links
        link_names = [ln.get('name') for ln in root.findall('link')]

        # Joints (keep every joint to preserve the tree; actuation handled via type)
        joints = root.findall('joint')
        parent_names, child_names, joint_types = [], [], []
        origins_xyz, origins_rpy, axes = [], [], []
        revolute_joint_names = []

        for j in joints:
            jname = j.get('name')
            jtype = j.get('type') or 'fixed'
            parent = j.find('parent').get('link')
            child  = j.find('child').get('link')
            origin = j.find('origin')
            xyz = [0.0, 0.0, 0.0] if origin is None else list(map(float, (origin.get('xyz') or "0 0 0").split()))
            rpy = [0.0, 0.0, 0.0] if origin is None else list(map(float, (origin.get('rpy') or "0 0 0").split()))
            axis_elem = j.find('axis')
            axis = [1.0, 0.0, 0.0] if axis_elem is None else list(map(float, (axis_elem.get('xyz') or "1 0 0").split()))

            parent_names.append(parent)
            child_names.append(child)
            joint_types.append(jtype)
            origins_xyz.append(xyz)
            origins_rpy.append(rpy)
            axes.append(axis)

            if jtype in ("revolute", "continuous"):
                revolute_joint_names.append(jname)

        return (link_names, parent_names, child_names, joint_types,
                origins_xyz, origins_rpy, axes, revolute_joint_names)

    @staticmethod
    def _parse_joint_names(urdf_path):
        # Edge-aligned joint names for mapping edge→dof
        root = ET.parse(urdf_path).getroot()
        return [j.get('name') for j in root.findall('joint')]

    @staticmethod
    def _topo_edge_order(L: int, parent_idx: torch.Tensor, child_idx: torch.Tensor) -> torch.Tensor:
        """
        Produce a deterministic parent→child order of edges for tree accumulation.
        Works for trees; if there are cycles (shouldn't be in URDF), behavior undefined.
        """
        E = parent_idx.numel()
        # find root: link that never appears as child
        all_links = torch.arange(L, dtype=torch.long)
        has_parent = torch.zeros(L, dtype=torch.bool)
        has_parent[child_idx] = True
        roots = all_links[~has_parent]
        root_link = int(roots[0].item()) if roots.numel() > 0 else 0

        # adjacency by edges
        children_by_parent = [[] for _ in range(L)]
        edges_by_parent = [[] for _ in range(L)]
        for e in range(E):
            p = int(parent_idx[e].item())
            children_by_parent[p].append(int(child_idx[e].item()))
            edges_by_parent[p].append(e)

        order = []
        stack = [root_link]
        visited_links = set([root_link])
        while stack:
            p = stack.pop()
            # process edges from p
            for e in edges_by_parent[p]:
                c = int(child_idx[e].item())
                order.append(e)
                if c not in visited_links:
                    visited_links.add(c)
                    stack.append(c)

        if len(order) != E:
            # Fallback to original order if something weird happened
            order = list(range(E))
        return torch.tensor(order, dtype=torch.long)
