import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class RBF_Variance(nn.Module):
    """
    Models the precision (inverse variance) of the decoder with an RBF network.
    beta(z) = W * v(z) + zeta
    where v_k(z) = exp(-lambda_k * ||z - c_k||²)
    """

    def __init__(self, d, D, K, zeta=1e-6):
        super(RBF_Variance, self).__init__()

        self.d = d  # latent dimension
        self.D = D  # input dimension
        self.K = K  # number of RBF centers
        self.zeta = zeta  # small constant to avoid division by zero

        # Centers and bandwidths (not trained, fixed by KMeans)
        self.register_buffer('centers', torch.zeros(K, d))
        self.register_buffer('lambdas', torch.ones(K))

        # Weights W > 0 (positivity constraint via softplus)
        self.W_raw = nn.Parameter(torch.randn(D, K))

    @property
    def W(self):
        return torch.nn.functional.softplus(self.W_raw)  # force W > 0

    def init_centers(self, Z, a=1.0):
        """
        Initializes centers with KMeans and bandwidths according to eq. 11
        Z : (N, d) tensor of encoded latent points
        a : curvature hyperparameter
        """
        Z_np = Z.detach().cpu().numpy()

        kmeans = KMeans(n_clusters=self.K, n_init=10)
        kmeans.fit(Z_np)

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.centers.copy_(centers)

        # Compute bandwidths according to eq. 11 of the paper
        # lambda_k = 0.5 * (a * (1/|C_k|) * sum ||z_j - c_k||²)^(-2)
        labels = kmeans.labels_
        lambdas = []

        for k in range(self.K):
            idx = labels == k
            if idx.sum() > 0:
                z_k = torch.tensor(Z_np[idx], dtype=torch.float32)
                c_k = centers[k]
                mean_dist = ((z_k - c_k) ** 2).sum(dim=1).mean().item()
                lam = 0.5 * (a * mean_dist) ** (-2)
            else:
                lam = 1.0
            lambdas.append(lam)

        self.lambdas.copy_(torch.tensor(lambdas, dtype=torch.float32))
        print(f"RBF centers initialized")

    def forward(self, z):
        """
        z : (N, d)
        returns log_var : (N, D) — log variance of the decoder
        """
        # v_k(z) = exp(-lambda_k * ||z - c_k||²)
        diff = z.unsqueeze(1) - self.centers.unsqueeze(0)  # (N, K, d)
        sq_dist = (diff ** 2).sum(dim=2)                   # (N, K)
        v = torch.exp(-self.lambdas * sq_dist)             # (N, K)

        # beta(z) = W * v(z) + zeta — precision
        beta = v @ self.W.T + self.zeta                    # (N, D)

        # variance = 1 / beta, log_var = -log(beta)
        log_var = -torch.log(beta)

        return log_var