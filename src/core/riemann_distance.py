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
            lambda z: model.decode(z)[0], z_single, vectorize=True
        )  # (1, D, 1, d)
        J = J[0, :, 0, :]  # (D, d)
        M = J.T @ J  # (d, d)
        all_M.append(M.detach())
    
    all_M = torch.stack(all_M, dim=0)  # (grid_size², d, d)
    return grid_z.detach(), all_M


def lookup_metric_batch(z_points, grid_z, all_M, grid_size=50):
    # Normalise z_points vers [0, grid_size-1]
    z_min = grid_z.min(dim=0).values
    z_max = grid_z.max(dim=0).values
    
    z_norm = (z_points - z_min) / (z_max - z_min) * (grid_size - 1)
    
    x0 = z_norm[:, 0].long().clamp(0, grid_size - 2)
    y0 = z_norm[:, 1].long().clamp(0, grid_size - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    # Poids bilinéaires
    wx = (z_norm[:, 0] - x0.float()).unsqueeze(1).unsqueeze(2)
    wy = (z_norm[:, 1] - y0.float()).unsqueeze(1).unsqueeze(2)

    all_M_grid = all_M.reshape(grid_size, grid_size, 2, 2)

    M00 = all_M_grid[x0, y0]
    M01 = all_M_grid[x0, y1]
    M10 = all_M_grid[x1, y0]
    M11 = all_M_grid[x1, y1]

    M = (1 - wx) * (1 - wy) * M00 + \
        (1 - wx) * wy       * M01 + \
        wx       * (1 - wy) * M10 + \
        wx       * wy       * M11

    return M


def riemannian_distance_batch(model, Z, centers, grid_z, all_M, steps=20, grid_size=50):
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

            M = lookup_metric_batch(z_mid, grid_z, all_M, grid_size)  # (N, d, d)

            # dz^T G dz vectorized for all points in the batch
            scalar = torch.bmm(
                dz.unsqueeze(1),               # (N, 1, d)
                torch.bmm(M, dz.unsqueeze(2))  # (N, d, 1)
            ).squeeze()                         # (N,)

            dist += torch.sqrt(torch.clamp(scalar, min=1e-10))

        dists[:, k] = dist

    return dists
