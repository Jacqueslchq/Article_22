import torch
import matplotlib.pyplot as plt
import numpy as np

def interpolate_latent(model, rbf_model, z_start, z_end, device, steps=10, geodesic_optim_steps=2000):
    """
    Interpolates between two latent points and decodes to image space.
    Returns linear and geodesic interpolations.
    """
    z1 = torch.tensor(z_start, dtype=torch.float32, device=device)
    z2 = torch.tensor(z_end, dtype=torch.float32, device=device)

    # Linear interpolation
    t = torch.linspace(0, 1, steps, device=device)
    linear_path = z1 + t.unsqueeze(1) * (z2 - z1)  # (steps, 2)

    # Geodesic interpolation
    path_init = linear_path.clone()
    path_inner = path_init[1:-1].clone().detach().requires_grad_(True)
    optimizer_geo = torch.optim.Adam([path_inner], lr=5e-3)

    rbf_model.eval()
    for _ in range(geodesic_optim_steps):
        full_path = torch.cat([z1.unsqueeze(0), path_inner, z2.unsqueeze(0)], dim=0)

        energy = torch.tensor(0.0, device=device)
        for i in range(len(full_path) - 1):
            dz = full_path[i+1] - full_path[i]
            z_mid = (full_path[i] + full_path[i+1]) / 2

            log_v = rbf_model(z_mid.unsqueeze(0))   # (1, D)
            weight = log_v.exp().mean()              # différentiable

            energy = energy + weight * (dz ** 2).sum()

        optimizer_geo.zero_grad()
        energy.backward()
        optimizer_geo.step()

    with torch.no_grad():
        geodesic_path = torch.cat([
            z1.unsqueeze(0), path_inner, z2.unsqueeze(0)
        ], dim=0)  # (steps, 2)

    # Decode both paths
    model.eval()
    with torch.no_grad():
        linear_imgs = model.decode(linear_path)[0]      # (steps, D)
        geodesic_imgs = model.decode(geodesic_path)[0]  # (steps, D)

    return linear_imgs.cpu(), geodesic_imgs.cpu()


def plot_interpolation(model, rbf_model, Z_np, labels_np, device, steps=8):
    """
    Pick one point from each class and interpolate.
    """
    # Take one representative point from each class (closest to the class center)
    for cls in np.unique(labels_np):
        idx = np.where(labels_np == cls)[0]
        center = Z_np[idx].mean(axis=0)
        dists = ((Z_np[idx] - center) ** 2).sum(axis=1)
        representative = Z_np[idx[dists.argmin()]]
        if cls == np.unique(labels_np)[0]:
            z_start = representative
        else:
            z_end = representative

    linear_imgs, geodesic_imgs = interpolate_latent(
        model, rbf_model, z_start, z_end, device, steps=steps
    )

    # Plot
    img_size = int(linear_imgs.shape[1] ** 0.5)  # 28 for MNIST

    fig, axes = plt.subplots(2, steps, figsize=(steps * 2, 4))

    for i in range(steps):
        # Linear
        axes[0, i].imshow(
            linear_imgs[i].reshape(img_size, img_size).numpy(),
            cmap='gray', vmin=0, vmax=1
        )
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Linear', fontsize=10, loc='left')

        # Geodesic
        axes[1, i].imshow(
            geodesic_imgs[i].reshape(img_size, img_size).numpy(),
            cmap='gray', vmin=0, vmax=1
        )
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Geodesic', fontsize=10, loc='left')

    plt.suptitle("Latent Space Interpolation: Linear vs Geodesic", fontsize=13)
    plt.tight_layout()
    plt.show()