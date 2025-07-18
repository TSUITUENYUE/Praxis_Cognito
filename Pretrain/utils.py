import torch
from torch_geometric.utils import dense_to_sparse



def build_edge_index(fk_model, end_effector_indices, device):
    link_names = fk_model.link_names
    num_links = len(link_names)
    num_nodes = num_links + 1  # +1 for object

    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    for joint_name, joint in fk_model.joint_map.items():
        if joint['parent'] in link_names and joint['child'] in link_names:
            p_idx = link_names.index(joint['parent'])
            c_idx = link_names.index(joint['child'])
            adj[p_idx, c_idx] = 1
            adj[c_idx, p_idx] = 1

    obj_index = num_links
    for ee_idx in end_effector_indices:
        adj[ee_idx, obj_index] = 1
        adj[obj_index, ee_idx] = 1

    edge_index_maybe_float, _ = dense_to_sparse(adj)
    return edge_index_maybe_float.long()