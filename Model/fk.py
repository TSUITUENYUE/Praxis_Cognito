import torch
import torch.nn as nn
from xml.etree import ElementTree as ET
from collections import defaultdict
import os
import sys

def vectorized_euler_to_rot_matrix(rpy):
    if rpy.dim() == 1:
        rpy = rpy.unsqueeze(0)
    bs = rpy.shape[0]
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)

    R = torch.zeros(bs, 3, 3, device=rpy.device)
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

def vectorized_axis_angle_to_rot_matrix(axis_angle):
    if axis_angle.dim() == 1:
        axis_angle = axis_angle.unsqueeze(0)
    bs = axis_angle.shape[0]
    angle = torch.norm(axis_angle, dim=1)
    mask = (angle == 0)
    axis = axis_angle / angle.unsqueeze(1).clamp(min=1e-6)
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    one_ca = 1 - ca
    kx, ky, kz = axis[:,0], axis[:,1], axis[:,2]

    R = torch.zeros(bs, 3, 3, device=axis_angle.device)
    R[:, 0, 0] = ca + kx*kx*one_ca
    R[:, 0, 1] = kx*ky*one_ca - kz*sa
    R[:, 0, 2] = kx*kz*one_ca + ky*sa
    R[:, 1, 0] = ky*kx*one_ca + kz*sa
    R[:, 1, 1] = ca + ky*ky*one_ca
    R[:, 1, 2] = ky*kz*one_ca - kx*sa
    R[:, 2, 0] = kz*kx*one_ca - ky*sa
    R[:, 2, 1] = kz*ky*one_ca + kx*sa
    R[:, 2, 2] = ca + kz*kz*one_ca

    eye = torch.eye(3, device=axis_angle.device).unsqueeze(0).repeat(bs, 1, 1)
    R[mask] = eye[mask]
    if bs == 1:
        return R.squeeze(0)
    return R

class FKModel(nn.Module):
    def __init__(self, urdf):
        super(FKModel, self).__init__()
        self.parse_urdf_file(urdf)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Precompute topological computation order (list of (parent, child, joint_name))
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

        # Precompute fixed tensors for each joint
        self.eye_4 = torch.eye(4, device=self.device)
        self.joint_idx = {name: i for i, name in enumerate(self.joint_names)}
        for joint_name, joint in self.joint_map.items():
            xyz = torch.tensor(joint['xyz'], device=self.device)
            rpy = torch.tensor(joint['rpy'], device=self.device)
            R = vectorized_euler_to_rot_matrix(rpy)
            t = xyz
            T_origin = self.eye_4.clone()
            T_origin[:3, :3] = R
            T_origin[:3, 3] = t
            joint['T_origin'] = T_origin
            if joint['type'] == 'revolute':
                joint['axis_tensor'] = torch.tensor(joint['axis'], device=self.device)
            else:
                joint['T_joint'] = self.eye_4.clone()

    def parse_urdf_file(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        self._parse_urdf_root(root)

    def _parse_urdf_root(self, root):
        self.link_names = [link.get('name') for link in root.findall('link')]
        self.joint_names = [joint.get('name') for joint in root.findall('joint') if joint.get('type') == 'revolute']
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
            self.joint_map[name] = {'type': jtype, 'parent': parent, 'child': child, 'xyz': xyz, 'rpy': rpy,
                                    'axis': axis}

    def forward(self, joint_angles):
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
        bs = joint_angles.shape[0]
        transforms = {'base': self.eye_4.unsqueeze(0).repeat(bs, 1, 1)}
        for parent, child, joint_name in self.computation_order:
            joint = self.joint_map[joint_name]
            T_origin = joint['T_origin'].unsqueeze(0).repeat(bs, 1, 1)
            if joint['type'] == 'revolute':
                angle = joint_angles[:, self.joint_idx[joint_name]]
                axis = joint['axis_tensor'].unsqueeze(0).repeat(bs, 1)
                axis_angle = axis * angle.unsqueeze(1)
                R_joint = vectorized_axis_angle_to_rot_matrix(axis_angle)
                T_joint = self.eye_4.unsqueeze(0).repeat(bs, 1, 1)
                T_joint[:, :3, :3] = R_joint
            else:
                T_joint = joint['T_joint'].unsqueeze(0).repeat(bs, 1, 1)
            T_child = torch.matmul(transforms[parent], torch.matmul(T_origin, T_joint))
            transforms[child] = T_child
        positions = torch.cat([transforms.get(link, self.eye_4.unsqueeze(0).repeat(bs, 1, 1))[:, :3, 3] for link in self.link_names], dim=1)
        if bs == 1:
            positions = positions.squeeze(0)
        return positions

if __name__ == '__main__':
    fk = FKModel("../Pretrain/urdfs/go2_description/urdf/go2_description.urdf")
    end_effector_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    end_effector_indices = [fk.link_names.index(name) for name in end_effector_names]
    print(len(fk.link_names))
    print("End effector indices:", end_effector_indices)
