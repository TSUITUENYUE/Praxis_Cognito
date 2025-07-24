import hydra
from omegaconf import DictConfig
from .fk import FKModel
import torch
import numpy as np
import xml.etree.ElementTree as ET


class Agent:
    def __init__(self, name, urdf, n_dofs, object_dim, joint_name, end_effector, init_angles=None):
        self.name = name
        self.urdf = urdf
        self.n_dofs = n_dofs
        self.object_dim = object_dim
        self.fk_model = FKModel(self.urdf)
        self.joint_name = joint_name  # Assuming this is a list of joint names
        self.init_angles = torch.tensor(init_angles) if init_angles is not None else torch.zeros(self.n_dofs)
        self.end_effector = end_effector
        self._get_joint_limit()


    def _get_joint_limit(self):
        # Parse URDF to get joint limits
        tree = ET.parse(self.urdf)
        root = tree.getroot()
        joint_limits_lower = []
        joint_limits_upper = []

        # Assuming joint_name is the list of joint names in order
        for name in self.joint_name:
            joint_elem = root.find(f".//joint[@name='{name}']")
            if joint_elem is not None:
                limit_elem = joint_elem.find('limit')
                if limit_elem is not None:
                    lower = float(limit_elem.get('lower', 0.0))
                    upper = float(limit_elem.get('upper', 0.0))
                    joint_limits_lower.append(lower)
                    joint_limits_upper.append(upper)
                else:
                    joint_limits_lower.append(0.0)  # Default if no limit
                    joint_limits_upper.append(0.0)
            else:
                joint_limits_lower.append(0.0)  # Default if joint not found
                joint_limits_upper.append(0.0)

        # Set as attribute (e.g., array of lowers, as in your example; or store both)
        self.joint_limits_lower = np.array(joint_limits_lower)
        self.joint_limits_upper = np.array(joint_limits_upper)