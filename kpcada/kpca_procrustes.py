"""KPCA with orthogonal Procrustes latent-basis alignment.

Source and target expression matrices are transposed to gene x sample, and
RBF kPCA (k=2, gamma = 1 / median of positive pairwise squared distances) is
run on each separately. The target embedding is then rotated (orthogonal
Procrustes, possibly with a reflection) so that the genes shared by both
datasets, used as anchors, land as close as possible to their source
coordinates. The rotation is applied to every target gene.
"""
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform


def median_heuristic_gamma(X):
    """gamma = 1 / median of the positive pairwise squared distances between rows of X."""
    d2 = pdist(X, metric="sqeuclidean")
    d2 = d2[d2 > 0]
    if d2.size == 0:
        raise ValueError("All rows are identical; cannot compute gamma.")
    return 1.0 / np.median(d2)


def rbf_kpca(X, n_components=2, gamma=None):
    """RBF kernel PCA on the rows of X (here: genes). Returns (scores, gamma, eigvals).

    scores = V * sqrt(lambda) from the double-centred kernel, shape (n_rows, n_components).
    """
    if gamma is None:
        gamma = median_heuristic_gamma(X)
    K = np.exp(-gamma * squareform(pdist(X, metric="sqeuclidean")))
    # double centring via row/column means (O(n^2), no n x n matrix products)
    rm = K.mean(axis=0)
    Kc = K - rm[None, :] - rm[:, None] + rm.mean()
    Kc = (Kc + Kc.T) / 2
    # only the top components are needed; eigsh avoids a full O(n^3) eigendecomposition
    if Kc.shape[0] > 500:
        vals, vecs = eigsh(Kc, k=n_components, which="LA")
    else:
        vals, vecs = np.linalg.eigh(Kc)
    order = np.argsort(vals)[::-1][:n_components]
    vals, vecs = vals[order], vecs[:, order]
    scores = vecs * np.sqrt(np.clip(vals, 0, None))
    return scores, gamma, vals


def orthogonal_procrustes(A, B):
    """Orthogonal R minimising ||A @ R - B||_F. Closed form via SVD of A.T @ B."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


class ProcrustesKPCA:
    """Separate kPCA on source and target, then Procrustes-align target to source.

    Parameters
    ----------
    n_components : int, number of kPCA components (default 2).
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, X_source, X_target, genes_source, genes_target):
        """
        X_source : (n_source_samples, n_source_genes)
        X_target : (n_target_samples, n_target_genes)  e.g. the target test set
        genes_source, genes_target : gene identifiers matching the columns.

        Returns (Z_source, Z_target_aligned), each (n_genes, n_components).
        """
        genes_source = np.asarray(genes_source)
        genes_target = np.asarray(genes_target)
        if X_source.shape[1] != len(genes_source) or X_target.shape[1] != len(genes_target):
            raise ValueError("Gene lists must match the number of columns in the matrices.")

        # samples x genes -> genes x samples; each gene is a point
        Gs = np.asarray(X_source, dtype=float).T
        Gt = np.asarray(X_target, dtype=float).T

        self.Z_source_, self.gamma_source_, self.eigvals_source_ = rbf_kpca(Gs, self.n_components)
        self.Z_target_, self.gamma_target_, self.eigvals_target_ = rbf_kpca(Gt, self.n_components)

        # anchors: genes present in both datasets
        src_index = {g: i for i, g in enumerate(genes_source)}
        anchor_src, anchor_tgt, anchors = [], [], []
        for j, g in enumerate(genes_target):
            if g in src_index:
                anchor_src.append(src_index[g])
                anchor_tgt.append(j)
                anchors.append(g)
        if len(anchors) < self.n_components:
            raise ValueError(f"Only {len(anchors)} shared genes; need at least {self.n_components}.")
        self.anchors_ = np.array(anchors)

        A_t = self.Z_target_[anchor_tgt]
        A_s = self.Z_source_[anchor_src]
        self.R_ = orthogonal_procrustes(A_t, A_s)

        # rotate every target gene, shared and unshared
        self.Z_target_aligned_ = self.Z_target_ @ self.R_
        self.anchor_rmse_before_ = float(np.sqrt(np.mean(np.sum((A_t - A_s) ** 2, axis=1))))
        self.anchor_rmse_after_ = float(np.sqrt(np.mean(np.sum((A_t @ self.R_ - A_s) ** 2, axis=1))))
        return self.Z_source_, self.Z_target_aligned_


if __name__ == "__main__":
    # self-check: target = rotated copy of source structure plus unshared genes
    rng = np.random.default_rng(0)
    n_s, n_g = 40, 200
    Xs = rng.normal(size=(n_s, n_g)) * np.linspace(0.2, 3, n_g)
    genes_s = np.array([f"g{i}" for i in range(n_g)])
    Xt = Xs[:30, :150] + 0.05 * rng.normal(size=(30, 150))
    genes_t = genes_s[:150]
    m = ProcrustesKPCA(2)
    m.fit_transform(Xs, Xt, genes_s, genes_t)
    print("R orthogonal:", np.allclose(m.R_.T @ m.R_, np.eye(2)))
    print("anchor RMSE before/after:", m.anchor_rmse_before_, m.anchor_rmse_after_)
