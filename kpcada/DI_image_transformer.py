"""DeepInsight-style image transformer driven by Procrustes-aligned kPCA coordinates.

pyDeepInsight scales each dataset's 2-D gene coordinates to the pixel grid on its own,
so the same gene lands on a different pixel in source and target. Here the source
embedding Zs and the aligned target embedding Zt are placed in ONE shared frame
(a single optional MBR rotation fitted on Zs, and a single min/max taken over both),
so an anchor gene falls on (nearly) the same pixel in both domains.

Usage
-----
    from kpcada import ImageTransformerDI
    tf = ImageTransformerDI(pixels=(224, 224)).fit(Zs, Zt)
    img_src = tf.transform(X_src_norm, domain="source")   # (n_samples, H, W, 3)
    img_tgt = tf.transform(X_tgt_norm, domain="target")

Columns of the matrices passed to `transform` must follow the same gene order as the
rows of Zs / Zt, and values should be scaled to [0, 1].
"""
from typing import Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull

from .pyDeepInsightv3 import ImageTransformer


class ImageTransformerDI(ImageTransformer):
    def __init__(self, discretization: str = "bin",
                 pixels: Union[int, Tuple[int, int]] = (224, 224),
                 rotate_to_mbr: bool = True) -> None:
        # feature extractor is unused: coordinates come from the aligned kPCA
        super().__init__(feature_extractor="pca", discretization=discretization, pixels=pixels)
        self._disc_name = discretization
        self.rotate_to_mbr = rotate_to_mbr
        self._coords_by_domain = {}

    def fit(self, Zs: np.ndarray, Zt: np.ndarray, plot: bool = False):
        """Zs: (n_source_genes, 2), Zt: (n_target_genes, 2) aligned embeddings."""
        Zs, Zt = np.asarray(Zs, float), np.asarray(Zt, float)
        if Zs.shape[1] != 2 or Zt.shape[1] != 2:
            raise ValueError("Zs and Zt must have 2 columns.")

        # one rotation, fitted on the source hull, applied to both domains
        rot = np.eye(2)
        if self.rotate_to_mbr:
            hull = Zs[ConvexHull(Zs).vertices]
            _, rot = self._minimum_bounding_rectangle(hull)
        Zs_r, Zt_r = Zs @ rot.T, Zt @ rot.T
        self.rotation_ = rot

        # one shared frame
        both = np.vstack([Zs_r, Zt_r])
        lo, hi = both.min(axis=0), both.max(axis=0)
        self.frame_ = (lo, hi)

        class _Framed(type(self)):
            @staticmethod
            def scale_coordinates(coords, dim_max):
                return (coords - lo) / (hi - lo) * np.asarray(dim_max)

        disc = {"bin": _Framed.coordinate_binning,
                "assignment": _Framed.coordinate_optimal_assignment,
                "lsa": _Framed.coordinate_optimal_assignment,
                "ags": _Framed.coordinate_heuristic_assignment}[self._disc_name]

        self._coords_by_domain = {
            "source": disc(Zs_r, self._pixels),
            "target": disc(Zt_r, self._pixels),
        }
        self._xrot_by_domain = {"source": Zs_r, "target": Zt_r}

        if plot:
            from matplotlib import pyplot as plt
            # source and target in their own panels, same axes (the shared frame used for the images)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
            for ax, Z, name, color in ((axes[0], Zs_r, "source", "tab:blue"),
                                       (axes[1], Zt_r, "target (aligned)", "tab:orange")):
                ax.scatter(*Z.T, s=2, alpha=0.4, c=color)
                ax.set_title(f"{name} (n={len(Z)} genes)")
                ax.set_aspect("equal", adjustable="box")
            plt.tight_layout()
            plt.show()
        return self

    def coords_for(self, domain: str) -> np.ndarray:
        return self._coords_by_domain[domain].copy()

    def transform(self, X: np.ndarray, domain: str = "source",
                  img_format: str = "rgb", empty_value: int = 0) -> np.ndarray:
        """X: (n_samples, n_genes_of_domain), scaled to [0, 1]. Returns one image per sample."""
        if domain not in self._coords_by_domain:
            raise ValueError("domain must be 'source' or 'target' (and fit() must be called first)")
        coords = self._coords_by_domain[domain]
        X = np.asarray(X, float)
        if X.shape[1] != coords.shape[0]:
            raise ValueError(f"X has {X.shape[1]} columns but the {domain} embedding has "
                             f"{coords.shape[0]} genes.")
        self._coords = coords
        return super().transform(X, img_format=img_format, empty_value=empty_value)

    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError("Call fit(Zs, Zt) then transform(X, domain=...).")


if __name__ == "__main__":
    # self-check: identical embeddings -> identical pixels for shared genes
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 2))
    tf = ImageTransformerDI(pixels=32).fit(Z, Z + 0.001 * rng.normal(size=Z.shape))
    X = rng.random((5, 300))
    a, b = tf.transform(X, "source", "scalar"), tf.transform(X, "target", "scalar")
    print(a.shape, "pixel agreement:", float(np.mean(a == b)))
