import hydra
from omegaconf import DictConfig
from .fk import FKModel
import torch
import torch.nn as nn
import numpy as np
import xml.etree.ElementTree as ET


class Agent(nn.Module):
    def __init__(self, name, urdf, n_dofs, object_dim, joint_name, end_effector, init_angles=None):
        super().__init__()
        self.name = name
        self.urdf = urdf
        self.n_dofs = n_dofs
        self.object_dim = object_dim
        self.joint_name = joint_name
        self.end_effector = end_effector


        fk = FKModel(self.urdf)
        self.fk_model = fk

        init = torch.zeros(self.n_dofs) if init_angles is None else torch.as_tensor(init_angles, dtype=torch.float32)
        self.register_buffer("init_angles", init)

        jl_lower, jl_upper = self._get_joint_limit_tensors()
        self.register_buffer("joint_limits_lower", jl_lower)
        self.register_buffer("joint_limits_upper", jl_upper)

    def _get_joint_limit_tensors(self):
        tree = ET.parse(self.urdf)
        root = tree.getroot()
        lowers, uppers = [], []
        for name in self.joint_name:
            joint_elem = root.find(f".//joint[@name='{name}']")
            if joint_elem is not None:
                limit_elem = joint_elem.find("limit")
                if limit_elem is not None:
                    lower = float(limit_elem.get("lower", 0.0))
                    upper = float(limit_elem.get("upper", 0.0))
                else:
                    lower = 0.0; upper = 0.0
            else:
                lower = 0.0; upper = 0.0
            lowers.append(lower); uppers.append(upper)
        # Return as torch tensors so they can be registered as buffers
        return torch.tensor(lowers, dtype=torch.float32), torch.tensor(uppers, dtype=torch.float32)