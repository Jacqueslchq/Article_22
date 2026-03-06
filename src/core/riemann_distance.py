import torch

def precompute_M(model, device, grid_size=50, z_range=(-3, 3)):
    xs = torch.linspace(z_range[0], z_range[1], grid_size, device=device)
    ys = torch.linspace(z_range[0], z_range[1], grid_size, device=device)
    
    grid_z = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).reshape(-1, 2)
    
    all_M = []
    
    # batch=1 to compute Jacobian for each point separately
    for i in range(len(grid_z)):
        z_single = grid_z[i:i+1].clone().detach().requires_grad_(True)
        J = torch.autograd.functional.jacobian(
            lambda z: model.decode(z), z_single, vectorize=True
        )  # (1, D, 1, d)
        J = J[0, :, 0, :]  # (D, d)
        M = J.T @ J  # (d, d)
        all_M.append(M.detach())
    
    all_M = torch.stack(all_M, dim=0)  # (grid_size², d, d)
    return grid_z.detach(), all_M


def lookup_metric_batch(z_points, grid_z, all_M):
    """
    z_points: (N, d)
    returns: (N, d, d)
    """
    # Nearest neighbor on all points of the grid
    diffs = torch.cdist(z_points, grid_z)  # (N, grid_size²)
    idx = diffs.argmin(dim=1)  # (N,)
    return all_M[idx]  # (N, d, d)


def riemannian_distance_batch(model, Z, centers, grid_z, all_M, steps=10):
    N, d = Z.shape
    K = centers.shape[0]
    device = Z.device
    dists = torch.zeros(N, K, device=device)
    path = torch.linspace(0, 1, steps, device=device)

    for k in range(K):
        z_path = Z.unsqueeze(0) + path.view(-1,1,1) * (centers[k] - Z).unsqueeze(0)
        dist = torch.zeros(N, device=device)

        for t in range(steps - 1):
            z_mid = (z_path[t] + z_path[t+1]) / 2  # (N, d)
            dz = z_path[t+1] - z_path[t]            # (N, d)

            M = lookup_metric_batch(z_mid, grid_z, all_M)  # (N, d, d)

            # dz^T G dz vectorized for all points in the batch
            scalar = torch.bmm(
                dz.unsqueeze(1),               # (N, 1, d)
                torch.bmm(M, dz.unsqueeze(2))  # (N, d, 1)
            ).squeeze()                         # (N,)

            dist += torch.sqrt(torch.clamp(scalar, min=1e-10))

        dists[:, k] = dist

    return dists