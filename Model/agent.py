import hydra
from omegaconf import DictConfig
from .fk import FKModel
import torch
import torch.nn as nn
import numpy as np
import xml.etree.ElementTree as ET
import math

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
        self._attach_link_collision_spheres()
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

    def _attach_link_collision_spheres(self):
        """
        Parse <collision><geometry> for each <link> and compute a per-link
        bounding-sphere radius that upper-bounds the collision shape.
        The order matches fk_model.link_names.
        """
        tree = ET.parse(self.urdf)
        root = tree.getroot()

        # Map link_name -> list of (type, params)
        # params: for sphere: {'radius': r}
        #         for cylinder: {'radius': r, 'length': l}
        #         for box: {'size': (sx,sy,sz)}
        link_geoms = {ln: [] for ln in self.fk_model.link_names}

        for link in root.findall('link'):
            lname = link.get('name')
            if lname not in link_geoms:
                continue
            # Prefer <collision>; fall back to <visual> if needed
            coll_nodes = link.findall('collision')
            if len(coll_nodes) == 0:
                coll_nodes = link.findall('visual')

            for coll in coll_nodes:
                geom = coll.find('geometry')
                if geom is None:
                    continue
                # sphere
                sph = geom.find('sphere')
                if sph is not None and sph.get('radius') is not None:
                    r = float(sph.get('radius'))
                    link_geoms[lname].append(('sphere', {'radius': r}))
                    continue
                # cylinder
                cyl = geom.find('cylinder')
                if cyl is not None and cyl.get('radius') is not None and cyl.get('length') is not None:
                    r = float(cyl.get('radius'));
                    L = float(cyl.get('length'))
                    link_geoms[lname].append(('cylinder', {'radius': r, 'length': L}))
                    continue
                # box
                box = geom.find('box')
                if box is not None and box.get('size') is not None:
                    sx, sy, sz = map(float, box.get('size').split())
                    link_geoms[lname].append(('box', {'size': (sx, sy, sz)}))
                    continue
                # otherwise skip

        # Convert to a single bounding-sphere radius per link
        radii = []
        for lname in self.fk_model.link_names:
            geos = link_geoms[lname]
            r_max = 0.0
            for gtype, params in geos:
                if gtype == 'sphere':
                    r = params['radius']
                elif gtype == 'cylinder':
                    # bounding sphere for a cylinder of radius r and length L
                    r = math.sqrt(params['radius'] ** 2 + (0.5 * params['length']) ** 2)
                elif gtype == 'box':
                    sx, sy, sz = params['size']
                    r = 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)
                else:
                    r = 0.0
                r_max = max(r_max, r)
            radii.append(r_max)

        self.register_buffer("link_bsphere_radius", torch.tensor(radii, dtype=torch.float32))