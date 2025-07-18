import hydra
from omegaconf import DictConfig
from .fk import FKModel
import torch


class Agent:
    def __init__(self, name, urdf, n_dofs, object_dim, joint_name, end_effector, init_angles=None):
        self.name = name
        self.urdf = urdf
        self.n_dofs = n_dofs
        self.object_dim = object_dim
        self.fk_model = FKModel(self.urdf)
        self.joint_name = joint_name
        self.init_angles = torch.tensor(init_angles) if init_angles is not None else torch.zeros(self.n_dofs)
        self.end_effector = end_effector