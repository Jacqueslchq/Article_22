from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from core.riemann_distance import riemannian_distance_batch, precompute_M


def match_labels(true_labels, pred_labels):
    unique = np.unique(true_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=unique)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: unique[row] for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[p] for p in pred_labels])


def kmeans_euclidean(Z, true_labels, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, n_init=30)
    pred_labels = kmeans.fit_predict(Z) # uses Euclidean distance by default

    pred_labels = match_labels(true_labels, pred_labels)
    
    f1 = f1_score(true_labels, pred_labels, average="macro")
    return f1, pred_labels


def kmeans_riemannian(Z, labels, model, n_clusters=3, n_iter=20):
    Z_cpu = Z.detach().cpu().numpy()
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(Z_cpu)

    Z_tensor = Z.to(next(model.parameters()).device)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=Z_tensor.device)

    grid_size=100
    # Precompute the M once for all iterations
    print("Precomputing M...")
    grid_z, all_M = precompute_M(model, Z_tensor.device, grid_size=grid_size)
    print("Done.")

    for it in range(n_iter):
        dists = riemannian_distance_batch(model, Z_tensor, centers, grid_z, all_M, grid_size=grid_size) # Approximate Riemannian distance
        assignments = dists.argmin(dim=1).cpu().numpy()

        for k in range(n_clusters):
            idx = assignments == k
            if idx.sum() > 0:
                centers[k] = Z_tensor[idx].mean(dim=0)

    assignments = match_labels(labels_np, assignments)
    f1 = f1_score(labels_np, assignments, average="macro")
    return f1, assignments


def plot_clustering_boundary(Z, labels, pred_labels, ax, title="Clustering"):
    Z = Z.astype(np.float32)
    labels = labels.astype(np.int64)
    pred_labels = pred_labels.astype(np.int64)

    for cls in np.unique(labels):
        idx = labels == cls
        ax.scatter(Z[idx, 0], Z[idx, 1], alpha=0.6, label=f"Digit {cls}")

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(Z, pred_labels)

    x_min, x_max = Z[:, 0].min() - 1, Z[:, 0].max() + 1
    y_min, y_max = Z[:, 1].min() - 1, Z[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    Z_pred = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z_pred, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.grid()
    ax.legend()