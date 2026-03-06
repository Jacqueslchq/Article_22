import torch
import matplotlib.pyplot as plt

def pairwise_dist2_torch(X, Y):
    """
    Compute pairwise squared Euclidean distance.

    X : (N x d)
    Y : (M x d)

    Output : (N x M)
    """

    XX = torch.sum(X**2, dim=1, keepdim=True)
    YY = torch.sum(Y**2, dim=1).unsqueeze(0)

    dist2 = XX + YY - 2 * (X @ Y.T)

    return torch.clamp(dist2, min=1e-12)

def extract_representation(model, train_loader=None, device="cpu"):
    model.eval
    latent_list = []
    label_list = []

    with torch.no_grad():

        for X, y in train_loader:

            X = X.to(device)

            _, _, Z, _ = model(X)

            latent_list.append(Z.cpu())
            label_list.append(y.cpu())

    Z, labels = torch.cat(latent_list), torch.cat(label_list)
    return Z, labels

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