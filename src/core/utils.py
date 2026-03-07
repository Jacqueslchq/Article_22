import torch
import matplotlib.pyplot as plt
from core.riemann_distance import precompute_M, lookup_metric_batch
import numpy as np

def extract_representation(model, train_loader=None, device="cpu"):
    model.eval
    latent_list = []
    label_list = []
    x_list = []

    with torch.no_grad():

        for X, y in train_loader:

            X = X.to(device)

            _, _, Z, _, _ = model(X)

            latent_list.append(Z.cpu())
            label_list.append(y.cpu())
            x_list.append(X.cpu())

    Z, labels = torch.cat(latent_list), torch.cat(label_list)
    X_train = torch.cat(x_list)
    return Z, labels, X_train

def plot_latent_space(Z, labels, max_per_class=1000):

    plt.figure(figsize=(5, 4))

    unique_classes = torch.unique(labels)

    for cls in unique_classes:

        idx = (labels == cls).nonzero(as_tuple=True)[0]

        # Randomly select up to max_per_class samples
        if len(idx) > max_per_class:
            perm = torch.randperm(len(idx))[:max_per_class]
            idx = idx[perm]

        Z_cls = Z[idx]

        plt.scatter(
            Z_cls[:, 0].cpu(),
            Z_cls[:, 1].cpu(),
            label=f"Class {cls.item()}",
            alpha=0.6,
            s=10
        )

    plt.legend()
    plt.title("Latent Space Embedding (1000 per class)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.show()

def sample_per_class(Z, labels, n_per_class=1000):

    indices = []

    unique_classes = torch.unique(labels)

    for cls in unique_classes:
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]

        if len(cls_idx) > n_per_class:
            perm = torch.randperm(len(cls_idx))[:n_per_class]
            cls_idx = cls_idx[perm]

        indices.append(cls_idx)

    indices = torch.cat(indices)

    return Z[indices], labels[indices]

def plot_input_space(train_loader, classes):
    examples, lab = next(iter(train_loader))

    plt.figure(figsize=(5,4))

    count = 0

    for cls in classes:

        idx = (lab == cls).nonzero()[0]

        plt.subplot(1,3,count+1)
        plt.imshow(examples[idx[0]].reshape(28,28), cmap="gray")
        plt.title(f"Digit {cls}")
        plt.axis("off")

        count += 1

    plt.tight_layout()
    plt.suptitle("Example images from the filtered MNIST dataset (input space)", y=.8)
    plt.show()

def plot_riemannian_metric(model, Z_np, device, grid_size=50, title="Riemannian Metric log det(M)"):
    z_range = (Z_np.min() - 0.5, Z_np.max() + 0.5)
    grid_z, all_M = precompute_M(model, device, grid_size=grid_size, z_range=z_range)
    
    det_M = torch.linalg.det(all_M).cpu().numpy()
    det_M = np.log(det_M + 1e-10).reshape(grid_size, grid_size)

    xs = np.linspace(z_range[0], z_range[1], grid_size)
    ys = np.linspace(z_range[0], z_range[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, det_M, levels=30, cmap="hot_r")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()

def plot_variance_comparison(model, rbf_model, Z_np, device, grid_size=100):
    
    z_range = (Z_np.min() - 0.5, Z_np.max() + 0.5)
    xs = torch.linspace(z_range[0], z_range[1], grid_size, device=device)
    ys = torch.linspace(z_range[0], z_range[1], grid_size, device=device)
    grid_z = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).reshape(-1, 2)

    xx, yy = np.meshgrid(
        np.linspace(z_range[0], z_range[1], grid_size),
        np.linspace(z_range[0], z_range[1], grid_size)
    )

    # --- Standard VAE variance ---
    model.eval()
    with torch.no_grad():
        _, log_var_std = model.decode(grid_z)
        variance_std = log_var_std.sum(dim=1).cpu().numpy().reshape(grid_size, grid_size)

    # --- RBF variance ---
    rbf_model.eval()
    with torch.no_grad():
        log_var_rbf = rbf_model(grid_z)
        variance_rbf = log_var_rbf.sum(dim=1).cpu().numpy().reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot standard
    im1 = axes[0].contourf(xx, yy, variance_std.T, levels=30, cmap="RdBu_r")
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_title("Standard Variance Estimate")
    axes[0].set_xlabel("z1")
    axes[0].set_ylabel("z2")

    # Plot RBF
    im2 = axes[1].contourf(xx, yy, variance_rbf.T, levels=30, cmap="RdBu_r")
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_title("Proposed Variance Estimate (RBF)")
    axes[1].set_xlabel("z1")
    axes[1].set_ylabel("z2")

    plt.tight_layout()
    plt.show()

def plot_geodesic(model, rbf_model, Z_np, labels_np, z_start, z_end, device, grid_size=100, geodesic_steps=30):

    z_range = (Z_np.min() - 0.5, Z_np.max() + 0.5)

    # Precompute riemannian metric on a grid
    grid_z, all_M = precompute_M(model, device, grid_size=grid_size, z_range=z_range)
    all_M = all_M.to(device)

    xx, yy = np.meshgrid(
        np.linspace(z_range[0], z_range[1], grid_size),
        np.linspace(z_range[0], z_range[1], grid_size)
    )

    # Heatmap variance RBF
    rbf_model.eval()
    xs = torch.linspace(z_range[0], z_range[1], grid_size, device=device)
    ys = torch.linspace(z_range[0], z_range[1], grid_size, device=device)
    grid_z_plot = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        log_var_rbf = rbf_model(grid_z_plot)
        variance_rbf = log_var_rbf.mean(dim=1).cpu().numpy().reshape(grid_size, grid_size)

    # Fix start and end points
    z1 = torch.tensor(z_start, dtype=torch.float32, device=device)
    z2 = torch.tensor(z_end, dtype=torch.float32, device=device)

    # Linear initialization of the path
    t = torch.linspace(0, 1, geodesic_steps, device=device)
    path_init = z1 + t.unsqueeze(1) * (z2 - z1)
    path_inner = path_init[1:-1].clone().detach().requires_grad_(True)
    optimizer_geo = torch.optim.Adam([path_inner], lr=5e-3)

    for _ in range(2000):
        full_path = torch.cat([z1.unsqueeze(0), path_inner, z2.unsqueeze(0)], dim=0)

        energy = torch.tensor(0.0, device=device)
        for i in range(len(full_path) - 1):
            dz = full_path[i+1] - full_path[i]          # (2,)
            z_mid = (full_path[i] + full_path[i+1]) / 2  # (2,)

            # Lookup metric at the midpoint using precomputed grid
            M = lookup_metric_batch(
                z_mid.unsqueeze(0), grid_z, all_M, grid_size
            )[0]  # (2, 2)

            energy = energy + torch.sqrt(
                torch.clamp(dz @ M @ dz, min=1e-10)
            )

        optimizer_geo.zero_grad()
        energy.backward()
        optimizer_geo.step()

    with torch.no_grad():
        geodesic_path = torch.cat([
            z1.unsqueeze(0), path_inner, z2.unsqueeze(0)
        ], dim=0).cpu().numpy()

    linear_path = np.array([
        z_start + t_val * (z_end - z_start)
        for t_val in np.linspace(0, 1, geodesic_steps)
    ])

    # Plot
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, variance_rbf.T, levels=30, cmap="RdBu_r")
    plt.colorbar()

    colors = ['green', 'orange']
    for i, cls in enumerate(np.unique(labels_np)):
        idx = labels_np == cls
        plt.scatter(Z_np[idx, 0], Z_np[idx, 1], alpha=0.3, s=5, color=colors[i], label=f"Class {cls}")

    plt.plot(geodesic_path[:, 0], geodesic_path[:, 1], 'r-', linewidth=2.5, label="Geodesic curve")
    plt.plot(linear_path[:, 0], linear_path[:, 1], 'k--', linewidth=2.5, label="Linear interpolation")

    plt.legend(fontsize=12)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Geodesic vs Linear Interpolation")
    plt.show()