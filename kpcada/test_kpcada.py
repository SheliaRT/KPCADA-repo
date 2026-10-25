"""Unit tests for kpcada.py (KSP loss, its gradient, the optimiser and the lambda rule).

Run from this folder:  python -m unittest test_kpcada -v
"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import KernelPCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kpcada  # noqa: E402

GAMMA = 0.05


def _target_kernel(X: np.ndarray) -> torch.Tensor:
    Xt = torch.tensor(X, dtype=torch.float64)
    return kpcada.center_kernel(kpcada.rbf_kernel(Xt, Xt, GAMMA))


def _record(lam, mmd, diverged=False):
    return {"lambda": lam, "final_mmd": float("nan") if diverged else mmd, "final_reg": 0.0,
            "diverged": diverged}


class TestKSP(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((60, 20))        # 60 genes x 20 samples
        self.K = _target_kernel(self.X)
        self.K_norm2 = (self.K * self.K).sum().item()

    def test_gradient_vanishes_at_kpca_solution(self):
        Z = torch.tensor(KernelPCA(n_components=2, kernel="rbf", gamma=GAMMA).fit_transform(self.X))
        grad, _ = kpcada._ksp_grad(Z, self.K, self.K_norm2)
        # relative to the size of the gradient at a random embedding of the same scale
        Z_rand = torch.randn_like(Z) * Z.std()
        grad_rand, _ = kpcada._ksp_grad(Z_rand, self.K, self.K_norm2)
        self.assertLess(grad.abs().max().item(), 1e-8 * max(1.0, grad_rand.abs().max().item()))

    def test_gradient_matches_autograd(self):
        torch.manual_seed(0)
        Z = torch.randn(60, 2, dtype=torch.float64, requires_grad=True)
        loss = kpcada.regularization_term(Z, self.K)
        loss.backward()
        grad, value = kpcada._ksp_grad(Z.detach(), self.K, self.K_norm2)
        torch.testing.assert_close(grad, Z.grad, rtol=1e-8, atol=1e-12)
        self.assertAlmostEqual(value, loss.item(), places=12)

    def test_invariant_to_rotation(self):
        torch.manual_seed(1)
        Z = torch.randn(60, 2, dtype=torch.float64)
        Q, _ = torch.linalg.qr(torch.randn(2, 2, dtype=torch.float64))   # orthogonal (rotation or reflection)
        self.assertAlmostEqual(kpcada.regularization_term(Z, self.K).item(),
                               kpcada.regularization_term(Z @ Q, self.K).item(), places=12)


class TestSelectLambda(unittest.TestCase):
    # mmd_init = 0.25, floor = 0.010 (lambda = 0), gain = 0.24
    # budgets: eps 0.05 -> 0.022, eps 0.10 -> 0.034, eps 0.20 -> 0.058
    TABLE = [(0.0, 0.010), (1e-3, 0.011), (1e-2, 0.025), (1e-1, 0.050), (1.0, 0.150), (10.0, None)]

    def _records(self):
        return [_record(l, m, diverged=m is None) for l, m in self.TABLE]

    def test_picks_largest_lambda_within_budget(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = kpcada.select_lambda(self._records(), mmd_init=0.25, epsilon=0.10)
        self.assertEqual(out["best_lambda"], 1e-2)
        self.assertAlmostEqual(out["mmd_floor"], 0.010)
        self.assertAlmostEqual(out["gain"], 0.24)
        self.assertEqual({e: r["lambda"] for e, r in out["sensitivity"].items()},
                         {0.05: 1e-3, 0.10: 1e-2, 0.20: 1e-1})
        self.assertEqual(out["best_lambda_mmd"], 0.0)

    def test_diverged_lambda_is_never_chosen(self):
        out = kpcada.select_lambda(self._records(), mmd_init=0.25, epsilon=10.0)   # huge budget
        self.assertEqual(out["best_lambda"], 1.0)                                  # 10.0 diverged

    def test_nonpositive_gain_raises(self):
        with self.assertRaises(ValueError):
            kpcada.select_lambda(self._records(), mmd_init=0.010)

    def test_warns_when_largest_grid_value_selected(self):
        records = [_record(0.0, 0.010), _record(1.0, 0.011)]
        with self.assertWarnsRegex(UserWarning, "largest grid value"):
            out = kpcada.select_lambda(records, mmd_init=0.25)
        self.assertEqual(out["best_lambda"], 1.0)

    def test_warns_when_only_zero_within_budget(self):
        records = [_record(0.0, 0.010), _record(1e-3, 0.200), _record(1e-2, 0.240)]
        with self.assertWarnsRegex(UserWarning, "No lambda other than 0"):
            out = kpcada.select_lambda(records, mmd_init=0.25)
        self.assertEqual(out["best_lambda"], 0.0)


class TestAdapt(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        Xs, Xt = rng.standard_normal((50, 15)), rng.standard_normal((40, 15)) + 0.5
        gs = np.array([f"g{i}" for i in range(50)])
        gt = np.array([f"g{i}" for i in range(10, 50)])
        self.prep = kpcada._prepare(Xs, Xt, gs, gt, n_components=2, kernel="rbf", kpca_gamma=None,
                                    mmd_gamma=0.5, auto_gamma=True, procrustes_init=False, reg_gamma=None,
                                    verbose=False)

    def test_backtracking_keeps_loss_nonincreasing(self):
        # far too large a step; a low floor so that every accepted step had to lower the loss
        out = kpcada._adapt(self.prep, 0.1, lr=1e4, max_epochs=200, verbose=False, lr_min=1e-6)
        self.assertFalse(out["diverged"])
        self.assertLess(out["lr_used"], 1e4)
        self.assertGreater(out["lr_used"], 1e-6)
        self.assertGreater(out["n_backtracks"], 0)
        h = np.array(out["loss_history"])
        self.assertTrue(np.all(np.diff(h) <= 0))

    def test_lr_never_below_floor(self):
        out = kpcada._adapt(self.prep, 0.1, lr=1e4, max_epochs=200, verbose=False)   # default floor lr * 1e-4
        self.assertGreaterEqual(out["lr_used"], 1.0)

    def test_stopping_rule(self):
        out = kpcada._adapt(self.prep, 0.1, lr=5.0, max_epochs=5000, verbose=False, tol=1e-3, window=20)
        self.assertTrue(out["converged"])
        self.assertLess(out["epochs_used"], 5000)
        self.assertEqual(len(out["loss_history"]), out["epochs_used"] + 1)
        ref, last = out["loss_history"][-21], out["loss_history"][-1]
        self.assertLessEqual(abs(ref - last), 1e-3 * abs(ref))

    def test_max_epochs_cap(self):
        out = kpcada._adapt(self.prep, 0.1, lr=5.0, max_epochs=7, verbose=False, tol=0.0)
        self.assertFalse(out["converged"])
        self.assertEqual(out["epochs_used"], 7)

    def test_reported_values_match_returned_embedding(self):
        out = kpcada._adapt(self.prep, 0.1, lr=5.0, max_epochs=30, verbose=False)
        ci_s, ci_t = self.prep["common_source_idx"], self.prep["common_target_idx"]
        mmd = kpcada.mmd_loss(self.prep["Z_s"][ci_s], out["Z_t"][ci_t], self.prep["mmd_gamma"]).item()
        reg = kpcada.regularization_term(out["Z_t"], self.prep["K_t"]).item()
        self.assertAlmostEqual(out["final_mmd"], mmd, places=5)
        self.assertAlmostEqual(out["final_reg"], reg, places=5)


if __name__ == "__main__":
    unittest.main()
