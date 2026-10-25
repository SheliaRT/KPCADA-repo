"""KPCA-DA: kernel-PCA domain adaptation (JAMIA manuscript methodology).

Pipeline (gene x sample matrices, genes are the points):
  1. Separate RBF kPCA on source and target gene vectors (k=2), gamma = 1 / median of the
     positive pairwise squared distances between gene vectors.  Scores are V * sqrt(Lambda).
  2. (optional) orthogonal Procrustes rotation of the target embedding onto the source one
     using the shared genes as anchors.
  3. Adapt the target embedding Z_t by gradient descent on
         L = MMD^2(Z_s,shared ; Z_t,shared) + lambda * L_KSP
     with MMD^2 = mean K_ss + mean K_tt - 2 mean K_st and the kernel structure-preservation loss
         L_KSP = ||K~_t - Z_t Z_t^T||_F^2 / ||K~_t||_F^2          (Eq. 5)
     K~_t being the centered RBF kernel of the original target gene vectors. Gradient of the regulariser
         4 lambda / ||K~_t||_F^2 * (Z_t Z_t^T - K~_t) Z_t         (Eq. 8)
     and the update  Z_t <- Z_t - lr * [grad MMD^2 + grad lambda L_KSP]  (Eq. 9).
     Unshared genes receive only the KSP gradient, so no gene is discarded.
     Stopping: relative change of the total loss over a 20-epoch window <= tol (1e-6), or max_epochs;
     a step that raises the loss is rejected and lr halved (down to a floor). Same settings for every lambda.
  4. MMD bandwidth: median heuristic on the pooled initial shared-gene embeddings (Supp. Note 1).
  5. lambda is selected label-free by an MMD^2 budget: the largest lambda within epsilon * gain of the grid
     floor (gain = MMD^2 before adaptation - lowest MMD^2 on the grid, lambda = 0 included).
"""
import warnings
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SOURCE_PATH = r"C:\Zmyfiles\codes\research\KPCA-DA\KPCA-DA\examples\data\TvsN\TCGA_BRCA_NvT.csv"
DEFAULT_TARGET_PATH = r"C:\Zmyfiles\codes\research\KPCA-DA\KPCA-DA\examples\data\TvsN\GSE45498_NvT.csv"


def load_expression_matrix(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and reshape the expression matrix from disk, returning data and gene names."""
    expr = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    expr = expr.transpose()
    expr.columns = expr.iloc[0]
    matrix = expr.iloc[1:, 0:-1].astype(float).values
    genes = expr.iloc[:, :-1].columns.to_numpy()
    return matrix.T, genes


def load_labels(csv_path: str) -> np.ndarray:
    """Extract sample labels from the last column of the expression CSV."""
    expr = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    expr = expr.transpose()
    expr.columns = expr.iloc[0]
    return expr.iloc[1:, -1].values


def standardize_datasets(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize source and target datasets independently."""
    X_source_scaled = StandardScaler().fit_transform(X_source)
    X_target_scaled = StandardScaler().fit_transform(X_target)
    return X_source_scaled, X_target_scaled


def find_common_gene_indices(genes_source: np.ndarray, genes_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the shared genes in each dataset, row-aligned: genes_source[i_s[k]] == genes_target[i_t[k]].

    Duplicated gene names keep their first occurrence.
    """
    _, index_source, index_target = np.intersect1d(
        np.asarray(genes_source).astype(str), np.asarray(genes_target).astype(str), return_indices=True
    )
    return index_source, index_target


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    pairwise_distances = torch.cdist(x, y) ** 2
    return torch.exp(-gamma * pairwise_distances)


def median_heuristic_gamma(X: np.ndarray) -> float:
    """gamma = 1 / median of the positive pairwise squared distances between the rows of X."""
    d2 = torch.pdist(torch.as_tensor(X, dtype=torch.float32)) ** 2
    d2 = d2[d2 > 0]
    if d2.numel() == 0:
        raise ValueError("All rows are identical; cannot compute the median-heuristic gamma.")
    return (1.0 / torch.median(d2).clamp(min=1e-8)).item()


def compute_median_gamma(Zs: torch.Tensor, Zt: torch.Tensor) -> float:
    """MMD bandwidth (Supp. Note 1): gamma = 1 / median(D^2) over the pooled shared-gene embeddings."""
    return median_heuristic_gamma(torch.cat([Zs, Zt], dim=0).numpy())


def mmd_loss(Zs_common: torch.Tensor, Zt_common: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Empirical squared MMD, empirical estimate: mean K_ss + mean K_tt - 2 mean K_st."""
    K_ss = rbf_kernel(Zs_common, Zs_common, gamma)
    K_tt = rbf_kernel(Zt_common, Zt_common, gamma)
    K_st = rbf_kernel(Zs_common, Zt_common, gamma)
    return K_ss.mean() + K_tt.mean() - 2 * K_st.mean()


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-centered kernel matrix H K H (H = I - 11^T / n), the K~_t of Eq. 5."""
    return K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()


def regularization_term(Z_t: torch.Tensor, K_t: torch.Tensor) -> torch.Tensor:
    """L_KSP (Eq. 5): ||K~_t - Z_t Z_t^T||_F^2 / ||K~_t||_F^2, K_t being the centered target kernel K~_t."""
    D = K_t - Z_t @ Z_t.T
    return (D * D).sum() / (K_t * K_t).sum()


def _ksp_grad(Z_t: torch.Tensor, K_t: torch.Tensor, K_norm2: float) -> Tuple[torch.Tensor, float]:
    """Closed-form gradient of L_KSP w.r.t. Z_t (Eq. 8, without the lambda factor) and the L_KSP value.

    d L_KSP / d Z_t = 4 / ||K~_t||_F^2 * (Z_t Z_t^T - K~_t) Z_t
    """
    D = torch.addmm(K_t, Z_t, Z_t.T, beta=-1)           # Z_t Z_t^T - K~_t, one n x n buffer
    ksp = (D.flatten() @ D.flatten()).item() / K_norm2
    return (4 / K_norm2) * (D @ Z_t), ksp


def _mmd_grad_wrt_target(Zs: torch.Tensor, Zt: torch.Tensor, gamma: float,
                         mean_Kss: float) -> Tuple[torch.Tensor, float]:
    """Closed-form d MMD^2 / d Z_t (Eq. 19) and the MMD^2 value, without autograd graphs.

    For k = exp(-gamma ||a-b||^2),  d k(a, z_j) / d z_j = 2 gamma k (a - z_j).
    """
    n, s = Zt.shape[0], Zs.shape[0]
    K_tt = rbf_kernel(Zt, Zt, gamma)
    K_st = rbf_kernel(Zs, Zt, gamma)
    mmd = mean_Kss + K_tt.mean().item() - 2 * K_st.mean().item()
    g_tt = (4 * gamma / n ** 2) * (K_tt @ Zt - K_tt.sum(1, keepdim=True) * Zt)
    g_st = (4 * gamma / (n * s)) * (K_st.T @ Zs - K_st.sum(0).unsqueeze(1) * Zt)
    return g_tt - g_st, mmd


def procrustes_rotation(Z_target_anchor: np.ndarray, Z_source_anchor: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||Z_target_anchor R - Z_source_anchor||_F (closed form via SVD)."""
    U, _, Vt = np.linalg.svd(Z_target_anchor.T @ Z_source_anchor)
    return U @ Vt


def _prepare(
    X_source: np.ndarray,
    X_target: np.ndarray,
    genes_source: np.ndarray,
    genes_target: np.ndarray,
    *,
    n_components: int,
    kernel: str,
    kpca_gamma: Optional[float],
    mmd_gamma: float,
    auto_gamma: bool,
    procrustes_init: bool,
    reg_gamma: Optional[Any],
    verbose: bool,
    Z_source_init: Optional[np.ndarray] = None,
    Z_target_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Everything that does not depend on lambda: separate kPCA, gammas, shared-gene indices, K_t.

    If Z_source_init / Z_target_init are given (n_genes x n_components, row-aligned with genes_*), the
    separate kPCA and the Procrustes step are skipped and these embeddings (e.g. from ProcrustesKPCA)
    are the starting point; X_target is then only used for K_t.
    """
    if genes_source is None or genes_target is None:
        raise ValueError("genes_source and genes_target must be provided for gene-based matching")
    common_source_idx, common_target_idx = find_common_gene_indices(genes_source, genes_target)
    if common_source_idx.size == 0:
        raise ValueError("No common genes found between the source and target datasets.")

    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)

    provided = Z_source_init is not None or Z_target_init is not None
    if provided:
        if Z_source_init is None or Z_target_init is None:
            raise ValueError("Pass both Z_source_init and Z_target_init, or neither.")
        Z_s_np = np.asarray(Z_source_init, dtype=float)
        Z_t_np = np.asarray(Z_target_init, dtype=float)
        if Z_s_np.shape[0] != X_source.shape[0] or Z_t_np.shape[0] != X_target.shape[0]:
            raise ValueError("Z_*_init rows must match the rows (genes) of X_source / X_target.")
        gamma_s = kpca_gamma
        gamma_t = kpca_gamma if kpca_gamma is not None else (
            median_heuristic_gamma(X_target) if reg_gamma is None else None)
        kpca_s = kpca_t = None
    else:
        # Separate kPCA per domain; gamma = 1 / median positive squared distance between gene vectors
        gamma_s = kpca_gamma if kpca_gamma is not None else median_heuristic_gamma(X_source)
        gamma_t = kpca_gamma if kpca_gamma is not None else median_heuristic_gamma(X_target)
        kpca_s = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma_s)
        kpca_t = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma_t)
        Z_s_np = kpca_s.fit_transform(X_source)
        Z_t_np = kpca_t.fit_transform(X_target)

    R = None
    if procrustes_init and not provided:
        R = procrustes_rotation(Z_t_np[common_target_idx], Z_s_np[common_source_idx])
        Z_t_np = Z_t_np @ R

    Z_s = torch.tensor(Z_s_np, dtype=torch.float32)
    Z_t0 = torch.tensor(Z_t_np, dtype=torch.float32)
    Zs_common = Z_s[common_source_idx]

    if auto_gamma:
        mmd_gamma = compute_median_gamma(Zs_common, Z_t0[common_target_idx])
        if verbose:
            print(f"Auto-computed MMD gamma (median heuristic): {mmd_gamma:.6f}")

    # K~_t (Eq. 5): centered RBF kernel of the ORIGINAL target gene vectors. Bandwidth: reg_gamma if given,
    # else the target kPCA bandwidth, so that Z_t Z_t^T is the rank-k approximation of K~_t before adaptation.
    gamma_reg = mmd_gamma if reg_gamma == "mmd" else (reg_gamma if reg_gamma is not None else gamma_t)
    if gamma_reg is None:
        gamma_reg = median_heuristic_gamma(X_target)
    X_t_tensor = torch.tensor(X_target, dtype=torch.float32)
    K_t = center_kernel(rbf_kernel(X_t_tensor, X_t_tensor, gamma_reg))
    mean_Kss = rbf_kernel(Zs_common, Zs_common, mmd_gamma).mean().item()

    return {
        "Z_s": Z_s, "Z_t0": Z_t0, "K_t": K_t, "K_norm2": (K_t * K_t).sum().item(), "mean_Kss": mean_Kss,
        "common_source_idx": common_source_idx, "common_target_idx": common_target_idx,
        "mmd_gamma": mmd_gamma, "reg_gamma": gamma_reg, "kpca_gamma_source": gamma_s, "kpca_gamma_target": gamma_t,
        "kpca_source": kpca_s, "kpca_target": kpca_t, "procrustes_R": R,
    }


def _loss_and_grad(prep: Dict[str, Any], Z_t: torch.Tensor,
                   lambda_reg: float) -> Tuple[float, float, float, torch.Tensor]:
    """Total loss, MMD^2, L_KSP and the gradient of L = MMD^2 + lambda * L_KSP at Z_t."""
    ci_s, ci_t = prep["common_source_idx"], prep["common_target_idx"]
    g_mmd, mmd = _mmd_grad_wrt_target(prep["Z_s"][ci_s], Z_t[ci_t], prep["mmd_gamma"], prep["mean_Kss"])
    g_ksp, reg = _ksp_grad(Z_t, prep["K_t"], prep["K_norm2"])
    grad = lambda_reg * g_ksp                     # Eq. 8: unshared genes move only through this term
    grad[ci_t] += g_mmd                           # Eq. 7: shared genes also feel the MMD force
    return mmd + lambda_reg * reg, mmd, reg, grad


def _adapt(prep: Dict[str, Any], lambda_reg: float, lr: float, max_epochs: int, verbose: bool,
           tol: float = 1e-6, window: int = 20, lr_min: Optional[float] = None) -> Dict[str, Any]:
    """Gradient descent on L = MMD^2 + lambda * L_KSP over the target embedding (Eqs. 6-9).

    Stops when the relative change of the total loss over the last `window` epochs is <= tol, or after
    max_epochs. Backtracking: a step that raises the loss (or is not finite) is rejected and lr is halved,
    down to lr_min (default lr * 1e-4); at the floor the step is taken anyway. loss_history[0] is the loss
    before adaptation, loss_history[e] the loss after epoch e.
    """
    lr_init = lr
    lr_floor = lr * 1e-4 if lr_min is None else lr_min
    Z_t = prep["Z_t0"].clone()
    loss, mmd, reg, grad = _loss_and_grad(prep, Z_t, lambda_reg)
    loss_history: List[float] = [loss]
    n_backtracks, epochs_used, converged = 0, 0, False

    for epoch in range(1, max_epochs + 1):
        while True:
            Z_new = Z_t - lr * grad                   # Eq. 9
            new = _loss_and_grad(prep, Z_new, lambda_reg)
            if (np.isfinite(new[0]) and new[0] <= loss) or lr <= lr_floor:
                break
            lr = max(lr / 2, lr_floor)
            n_backtracks += 1
        if not np.isfinite(new[0]):
            if verbose:
                print(f"Diverged at epoch {epoch} with lr at its floor ({lr:.3g}); stopping.")
            return {"Z_t": Z_t, "loss_history": loss_history, "final_mmd": float("nan"),
                    "final_reg": float("nan"), "final_loss": float("nan"), "diverged": True,
                    "lr_init": lr_init, "lr_used": lr, "epochs_used": epoch - 1, "converged": False,
                    "n_backtracks": n_backtracks}
        Z_t = Z_new
        loss, mmd, reg, grad = new
        loss_history.append(loss)
        epochs_used = epoch
        if verbose and epoch % max(1, max_epochs // 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}, MMD: {mmd:.6f}, Reg: {reg:.4f}, lr: {lr:.3g}")
        if epoch >= window:
            ref = loss_history[-1 - window]
            if abs(ref - loss) <= tol * max(abs(ref), 1e-12):
                converged = True
                break

    if verbose:
        print(f"Stopped after {epochs_used} epochs ({'converged' if converged else 'max_epochs reached'}), "
              f"lr {lr_init:.3g} -> {lr:.3g}, {n_backtracks} backtracks")
    # mmd / reg are evaluated at the returned Z_t, so MMD, regulariser and total share one endpoint
    return {
        "Z_t": Z_t, "loss_history": loss_history,
        "final_mmd": mmd, "final_reg": reg, "final_loss": loss, "diverged": False,
        "lr_init": lr_init, "lr_used": lr, "epochs_used": epochs_used, "converged": converged,
        "n_backtracks": n_backtracks,
    }


def fit_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    genes_source: Optional[np.ndarray] = None,
    genes_target: Optional[np.ndarray] = None,
    *,
    n_components: int = 2,
    kernel: str = "rbf",
    kpca_gamma: Optional[float] = None,
    mmd_gamma: float = 0.5,
    auto_gamma: bool = True,
    lambda_reg: float = 0.001,
    lr: float = 0.001,
    max_epochs: int = 500,
    tol: float = 1e-6,
    window: int = 20,
    lr_min: Optional[float] = None,
    procrustes_init: bool = False,
    reg_gamma: Optional[Any] = None,
    verbose: bool = False,
    Z_source_init: Optional[np.ndarray] = None,
    Z_target_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Align source and target gene embeddings (inputs are genes x samples).

    kpca_gamma=None uses the median heuristic per domain; auto_gamma=True sets the MMD bandwidth from
    the initial shared-gene embeddings (replacing mmd_gamma). procrustes_init=True rotates the
    target embedding onto the source one (shared genes as anchors) before the MMD optimisation.
    Stopping rule and backtracking (max_epochs, tol, window, lr_min): see _adapt.
    """
    prep = _prepare(
        X_source, X_target, genes_source, genes_target, n_components=n_components, kernel=kernel,
        kpca_gamma=kpca_gamma, mmd_gamma=mmd_gamma, auto_gamma=auto_gamma,
        procrustes_init=procrustes_init, reg_gamma=reg_gamma, verbose=verbose,
        Z_source_init=Z_source_init, Z_target_init=Z_target_init,
    )
    out = _adapt(prep, lambda_reg, lr, max_epochs, verbose, tol=tol, window=window, lr_min=lr_min)
    return {
        "lr_used": out["lr_used"], "epochs_used": out["epochs_used"], "converged": out["converged"],
        "diverged": out["diverged"],
        "Z_s": prep["Z_s"].numpy(),
        "Z_t": out["Z_t"].numpy(),
        "Z_t_initial": prep["Z_t0"].numpy(),
        "common_source_idx": prep["common_source_idx"],
        "common_target_idx": prep["common_target_idx"],
        "loss_history": out["loss_history"],
        "final_mmd": out["final_mmd"], "final_reg": out["final_reg"], "final_loss": out["final_loss"],
        "kpca_model": prep["kpca_target"], "kpca_model_source": prep["kpca_source"],
        "common_genes_count": len(prep["common_source_idx"]),
        "mmd_gamma": prep["mmd_gamma"],
        "kpca_gamma_source": prep["kpca_gamma_source"], "kpca_gamma_target": prep["kpca_gamma_target"],
        "procrustes_R": prep["procrustes_R"], "reg_gamma": prep["reg_gamma"],
    }


class KPCA_DA:
    """Kernel PCA domain adaptation: separate kPCA, optional Procrustes, MMD + KSP-regularised adaptation."""

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        kpca_gamma: Optional[float] = None,
        mmd_gamma: float = 0.5,
        auto_gamma: bool = True,
        lambda_reg: float = 0.001,
        lr: float = 0.001,
        max_epochs: int = 500,
        tol: float = 1e-6,
        window: int = 20,
        lr_min: Optional[float] = None,
        procrustes_init: bool = False,
        reg_gamma: Optional[Any] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        n_components : number of kPCA components (k)
        kernel : kernel for kPCA
        kpca_gamma : kPCA gamma; None -> 1 / median positive pairwise squared distance, per domain
        mmd_gamma : MMD bandwidth (ignored when auto_gamma=True)
        auto_gamma : set the MMD bandwidth by the median heuristic on the initial shared-gene embeddings
        lambda_reg : weight of the KSP term
        lr : initial gradient-descent step size; halved when a step raises the loss, down to lr_min
            (None -> lr * 1e-4)
        max_epochs, tol, window : stop when the relative change of the total loss over `window` epochs is
            <= tol, or after max_epochs. The same settings are used for every lambda.
        procrustes_init : rotate the target embedding onto the source one before adaptation
        reg_gamma : bandwidth of K~_t in the KSP term; None -> target kPCA gamma; "mmd" -> the MMD gamma; or a float.
            With the target kPCA gamma, Z_t Z_t^T is the best rank-k approximation of K~_t before adaptation.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.kpca_gamma = kpca_gamma
        self.mmd_gamma = mmd_gamma
        self.auto_gamma = auto_gamma
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.max_epochs = max_epochs
        self.tol = tol
        self.window = window
        self.lr_min = lr_min
        self.procrustes_init = procrustes_init
        self.reg_gamma = reg_gamma
        self.verbose = verbose

        self.kpca_model = None          # target kPCA
        self.kpca_model_source = None
        self.Z_s_ = None
        self.Z_t_ = None
        self.common_source_idx_ = None
        self.common_target_idx_ = None
        self.loss_history_ = None
        self.final_mmd_ = None

    def fit_transform(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        genes_source: Optional[np.ndarray] = None,
        genes_target: Optional[np.ndarray] = None,
        Z_source_init: Optional[np.ndarray] = None,
        Z_target_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and return (Z_s, adapted Z_t). X_source, X_target are genes x samples.

        Z_source_init / Z_target_init: optional precomputed (Procrustes-aligned) gene embeddings to adapt
        instead of running the internal kPCA.
        """
        results = fit_transform(
            X_source=X_source, X_target=X_target,
            genes_source=genes_source, genes_target=genes_target,
            n_components=self.n_components, kernel=self.kernel, kpca_gamma=self.kpca_gamma,
            mmd_gamma=self.mmd_gamma, auto_gamma=self.auto_gamma, lambda_reg=self.lambda_reg,
            lr=self.lr, max_epochs=self.max_epochs, tol=self.tol, window=self.window, lr_min=self.lr_min,
            procrustes_init=self.procrustes_init, reg_gamma=self.reg_gamma,
            verbose=self.verbose, Z_source_init=Z_source_init, Z_target_init=Z_target_init,
        )
        self.lr_used_ = results["lr_used"]
        self.epochs_used_ = results["epochs_used"]
        self.converged_ = results["converged"]
        self.Z_s_ = results["Z_s"]
        self.Z_t_ = results["Z_t"]
        self.Z_t_initial_ = results["Z_t_initial"]
        self.common_source_idx_ = results["common_source_idx"]
        self.common_target_idx_ = results["common_target_idx"]
        self.loss_history_ = results["loss_history"]
        self.final_mmd_ = results["final_mmd"]
        self.mmd_gamma_ = results["mmd_gamma"]
        self.kpca_model = results["kpca_model"]
        self.kpca_model_source = results["kpca_model_source"]
        return self.Z_s_, self.Z_t_

    def tune_lambda(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        genes_source: np.ndarray,
        genes_target: np.ndarray,
        lambda_values: List[float] = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
        epsilon: float = 0.10,
        sensitivity_epsilons: Tuple[float, ...] = (0.05, 0.10, 0.20),
        Z_source_init: Optional[np.ndarray] = None,
        Z_target_init: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Label-free lambda sweep with an MMD^2 budget; sets self.lambda_reg and pins the MMD gamma.

        The selected lambda is the largest non-diverged one whose MMD^2 stays within epsilon * gain of the grid
        floor (see the module-level tune_lambda). Pass only the source and the target ADAPTATION data (never the held-out target test set).
        """
        summary = tune_lambda(
            X_source, X_target, genes_source, genes_target, lambda_values,
            epsilon=epsilon, sensitivity_epsilons=sensitivity_epsilons,
            n_components=self.n_components, kernel=self.kernel, kpca_gamma=self.kpca_gamma,
            mmd_gamma=self.mmd_gamma, auto_gamma=self.auto_gamma, lr=self.lr, max_epochs=self.max_epochs,
            tol=self.tol, window=self.window, lr_min=self.lr_min, procrustes_init=self.procrustes_init, reg_gamma=self.reg_gamma, verbose=self.verbose,
            Z_source_init=Z_source_init, Z_target_init=Z_target_init,
        )
        self.lambda_reg = summary["best_lambda"]
        self.mmd_gamma = summary["mmd_gamma"]
        self.auto_gamma = False
        return summary

    def transform(self, X: np.ndarray, domain: str = "target") -> np.ndarray:
        """Out-of-sample kPCA projection of new gene vectors (rows) with the fitted source/target kPCA.

        This is the plain kPCA projection; it does not apply the MMD adaptation.
        """
        model = self.kpca_model if domain == "target" else self.kpca_model_source
        if model is None:
            raise ValueError("Model has not been fitted yet. Call fit_transform first.")
        return model.transform(X)


def tune_lambda(
    X_source: np.ndarray,
    X_target: np.ndarray,
    genes_source: np.ndarray,
    genes_target: np.ndarray,
    lambda_values: List[float] = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
    epsilon: float = 0.10,
    sensitivity_epsilons: Tuple[float, ...] = (0.05, 0.10, 0.20),
    *,
    n_components: int = 2,
    kernel: str = "rbf",
    kpca_gamma: Optional[float] = None,
    mmd_gamma: float = 0.5,
    auto_gamma: bool = True,
    lr: float = 0.001,
    max_epochs: int = 500,
    tol: float = 1e-6,
    window: int = 20,
    lr_min: Optional[float] = None,
    procrustes_init: bool = False,
    reg_gamma: Optional[Any] = None,
    verbose: bool = False,
    Z_source_init: Optional[np.ndarray] = None,
    Z_target_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sweep lambda and report final MMD^2, KSP term and total loss at one endpoint per value.

    The kPCA embeddings, bandwidths and K_t are computed once and shared by every lambda; every lambda runs
    with the same gamma, initial lr, lr floor, tol, window and max_epochs. lambda = 0 is always added to the
    grid. The lr and number of epochs each run actually used are in the table (lr_used, epochs_used).
    Selection: see select_lambda. Call with the source and the target adaptation subset only.

    Returns the select_lambda dict plus "results" (list of dicts), "results_df", "mmd_gamma", "reg_gamma".
    """
    prep = _prepare(
        X_source, X_target, genes_source, genes_target, n_components=n_components, kernel=kernel,
        kpca_gamma=kpca_gamma, mmd_gamma=mmd_gamma, auto_gamma=auto_gamma,
        procrustes_init=procrustes_init, reg_gamma=reg_gamma, verbose=verbose,
        Z_source_init=Z_source_init, Z_target_init=Z_target_init,
    )
    lambda_values = sorted({float(l) for l in lambda_values} | {0.0})
    with torch.no_grad():
        mmd_init = mmd_loss(prep["Z_s"][prep["common_source_idx"]],
                            prep["Z_t0"][prep["common_target_idx"]], prep["mmd_gamma"]).item()
    records = []
    for lam in lambda_values:
        if verbose:
            print(f"\n--- lambda = {lam} ---")
        out = _adapt(prep, lam, lr, max_epochs, verbose, tol=tol, window=window, lr_min=lr_min)
        records.append({"lambda": lam, "final_mmd": out["final_mmd"], "final_reg": out["final_reg"],
                        "final_loss": out["final_loss"], "diverged": out["diverged"],
                        "lr_used": out["lr_used"], "epochs_used": out["epochs_used"],
                        "converged": out["converged"], "n_backtracks": out["n_backtracks"]})
        if verbose:
            print(f"  MMD={out['final_mmd']:.6f}  Reg={out['final_reg']:.4f}  Loss={out['final_loss']:.6f}"
                  f"  lr_used={out['lr_used']:.3g}  epochs_used={out['epochs_used']}")
    summary = select_lambda(records, mmd_init, epsilon, sensitivity_epsilons)
    summary.update({
        "results": records,
        "results_df": pd.DataFrame(records),
        "mmd_gamma": prep["mmd_gamma"],
        "reg_gamma": prep["reg_gamma"],
    })
    return summary


def select_lambda(records: List[Dict[str, Any]], mmd_init: float, epsilon: float = 0.10,
                  sensitivity_epsilons: Tuple[float, ...] = (0.05, 0.10, 0.20)) -> Dict[str, Any]:
    """Label-free lambda choice by an MMD^2 budget (never tune epsilon on test accuracy).

    records: one dict per lambda with "lambda", "final_mmd", "diverged" (and optionally "final_reg").
        gain   = MMD^2_init - MMD^2_floor      (init: before adaptation; floor: lowest MMD^2 on the grid)
        budget = MMD^2_floor + epsilon * gain
        best_lambda = the largest non-diverged lambda with MMD^2(lambda) <= budget
    Raises ValueError if gain <= 0 (adaptation did not lower MMD^2 on any lambda). Warns if the choice is the
    largest grid value (the grid may be too narrow) or if no lambda other than 0 is within budget.
    Adds a "within_budget" flag (at epsilon) to each record.

    Returns "best_lambda", "selected" (its record), "mmd_init", "mmd_floor", "gain", "epsilon", "budget",
    "sensitivity" ({eps: record}), "best_lambda_mmd" (the lowest-MMD^2 lambda, kept for reference).
    """
    finite = [r for r in records if not r["diverged"]]
    if not finite:
        raise RuntimeError("Every lambda diverged; lower lr or lambda.")
    best = min(finite, key=lambda r: r["final_mmd"])
    mmd_floor = best["final_mmd"]
    gain = mmd_init - mmd_floor
    if gain <= 0:
        raise ValueError(f"gain = MMD^2_init - MMD^2_floor = {mmd_init:.6g} - {mmd_floor:.6g} <= 0: no lambda "
                         "lowered MMD^2 below its pre-adaptation value, so the budget rule is undefined. "
                         "Check lr / max_epochs / gamma.")

    def choose(eps: float) -> Dict[str, Any]:
        budget = mmd_floor + eps * gain
        return max((r for r in finite if r["final_mmd"] <= budget), key=lambda r: r["lambda"])

    selected = choose(epsilon)
    budget = mmd_floor + epsilon * gain
    for r in records:
        r["within_budget"] = (not r["diverged"]) and r["final_mmd"] <= budget
    if selected["lambda"] == max(r["lambda"] for r in records):
        warnings.warn(f"Selected lambda {selected['lambda']} is the largest grid value; extend the grid upward.")
    if not any(r["within_budget"] and r["lambda"] > 0 for r in records):
        warnings.warn("No lambda other than 0 is within the MMD^2 budget; the KSP term is effectively off. "
                      "Refine the grid near 0 or check the step size.")
    return {
        "best_lambda": selected["lambda"],
        "selected": selected,
        "mmd_init": mmd_init, "mmd_floor": mmd_floor, "gain": gain, "epsilon": epsilon, "budget": budget,
        "sensitivity": {e: choose(e) for e in sensitivity_epsilons},
        "best_lambda_mmd": best["lambda"],
    }


def main() -> None:
    # Load data with gene names (genes x samples)
    X_source, genes_source = load_expression_matrix(DEFAULT_SOURCE_PATH)
    X_target, genes_target = load_expression_matrix(DEFAULT_TARGET_PATH)

    model = KPCA_DA(n_components=2, kernel="rbf", verbose=True)
    Z_s, Z_t = model.fit_transform(X_source, X_target, genes_source, genes_target)
    print(f"\nCommon genes found: {len(model.common_source_idx_)}")
    print(f"Transformed source shape: {Z_s.shape}")
    print(f"Adapted target shape: {Z_t.shape}")
    print(f"Final MMD^2: {model.final_mmd_:.6f}")


if __name__ == "__main__":
    main()
