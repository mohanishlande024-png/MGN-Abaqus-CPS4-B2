"""
KtMGN - Run B2: Run B1 with ONE change, the training target.

    B1 learns   z = (stress - mu) / s            one mu, s over all nodes of all loads
    B2 learns   z = (Kt - mu) / s,  Kt = stress / load (nominal traction)
                and predicts  stress = load x Kt

WHY
    The plate is linear elastic, so stress = load x Kt(x, y) and Kt does not
    depend on the load. In B1 the network has to build the "x load" itself and
    its error is about the same size in MPa at every load, so R2 falls at low
    loads (the field spread shrinks with load^2, the error does not). In B2 the
    multiplication by the load is done exactly, outside the network: the
    network's error in Kt is multiplied by the load, so for one geometry the
    R2 numerator and denominator both scale with load^2 and R2 no longer
    depends on the load.

    The R2 formula does NOT change: R2 = 1 - sum(err^2) / sum((FE - mean FE)^2)
    on the stress of one file. For one file, R2 on stress = R2 on Kt (load^2
    cancels), so B1 and B2 are scored exactly the same way.

EVERYTHING ELSE IS B1
    Same graph, same 20% random edges re-drawn every epoch, same 14 anchor
    numbers in every layer, same model size and training step. The load is
    still one of the 15 global inputs (load_input=True), so B1 vs B2 is a clean
    one-change comparison. load_input=False zeroes that column (the network
    then cannot see the load at all, and Kt is the same at every load) - an
    optional second ablation.

CHECKPOINTS
    multi_geometry_mgn.pt carries ck["b1"] (the B1 settings) AND ck["b2"]
    (target, load_input); mgn_best_val.pt carries config["target"] = "kt".
    scripts/eval_unseen_v2.py and scripts/check_fit.py detect both and rebuild
    the right model. A B1 checkpoint is never read as B2 or the other way round.
"""

import numpy as np
import torch

from .anchor_trainer import AnchorMGN


TARGET = "kt"


def load_of(fem):
    """The nominal traction of one graph (fem.metadata['load']), as a float > 0."""
    L = float(fem.metadata["load"])
    if not L > 0:
        raise ValueError("'%s': load must be > 0 for a Kt target (got %r)" % (getattr(fem, "geometry", "?"), L))
    return L


def kt_targets(cases):
    """Concatenated Kt = von Mises / load of a list of graphs (the B2 training target)."""
    return np.concatenate([np.asarray(c.von_mises, dtype=np.float64).ravel() / load_of(c) for c in cases])


class KtMGN(AnchorMGN):

    def __init__(self, *, load_input=True, **kwargs):
        super().__init__(**kwargs)
        self.load_input = bool(load_input)

    def config(self):
        c = super().config()
        c.update(target=TARGET, load_input=self.load_input)
        return c

    # ------------------------------------------------------------------ inputs

    def _fem_to_graph(self, fem_list):
        node_types, coords, edge_index, glob = super()._fem_to_graph(fem_list)
        if not self.load_input and glob is not None and "load" in self.global_features:
            j = self.global_features.index("load")
            glob = glob.clone()
            glob[:, j] = 0.0            # constant column: z-scored to 0 (AnchorMGN leaves flat columns unscaled)
        return node_types, coords, edge_index, glob

    # ------------------------------------------------------------------ target

    def fit(self, fem, y=None, batch_size=1):
        """Same as MGN.fit, but the target is converted to Kt first (y = stress, or None to use fem.von_mises)."""
        fems = fem if isinstance(fem, list) else [fem]
        if y is None:
            y_kt = kt_targets(fems)
        else:
            y = np.asarray(y, dtype=np.float64).ravel()
            parts, k = [], 0
            for f in fems:
                n = len(f.coordinates)
                parts.append(y[k:k + n] / load_of(f))
                k += n
            y_kt = np.concatenate(parts)
        return super().fit(fem, y_kt, batch_size=batch_size)

    def predict_kt(self, fem):
        """The network's own output, de-normalised: Kt at every node."""
        if isinstance(fem, list):
            return np.concatenate([np.asarray(self.predict_kt(f)).ravel() for f in fem])
        return np.asarray(super().predict(fem)).ravel()

    def predict(self, fem):
        """Stress = load x Kt, so every script that calls predict() gets MPa (or psi) as before."""
        if isinstance(fem, list):
            return np.concatenate([np.asarray(self.predict(f)).ravel() for f in fem])
        return self.predict_kt(fem) * load_of(fem)

    # ------------------------------------------------------------------ save / load

    def save(self, filepath):
        super().save(filepath)                              # writes ck["b1"] = config() (now incl. target)
        ck = torch.load(filepath, map_location="cpu", weights_only=False)
        ck["b2"] = dict(target=TARGET, load_input=self.load_input)
        torch.save(ck, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        ck = torch.load(filepath, map_location="cpu", weights_only=False)
        if "b2" not in ck:
            raise ValueError("%s is not a B2 (Kt target) checkpoint - load it with AnchorMGN.load" % filepath)
        mgn = super(KtMGN, cls).load(filepath, device)
        mgn.load_input = bool(ck["b2"].get("load_input", True))
        print("  B2: target Kt = stress / load, load input %s" % ("on" if mgn.load_input else "OFF"))
        return mgn
