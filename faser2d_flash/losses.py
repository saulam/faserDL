from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CylindricalConsistencyLoss(nn.Module):
    def __init__(self, stats: dict[str, Any], huber_delta: float = 1.0):
        super().__init__()
        self.huber_delta = float(huber_delta)
        self.register_buffer(
            "s_vis_xyz", torch.tensor(stats["vis"]["s_xyz"], dtype=torch.float32).view(1, 3)
        )
        self.register_buffer(
            "s_jet_xyz", torch.tensor(stats["jet"]["s_xyz"], dtype=torch.float32).view(1, 3)
        )
        self.register_buffer(
            "s_lep_xyz", torch.tensor(stats["lep"]["s_xyz"], dtype=torch.float32).view(1, 3)
        )
        self.tau_pt_vis = float(stats["vis"]["tau_pt"])
        self.tau_pt_jet = float(stats["jet"]["tau_pt"])
        self.tau_mag_lep = float(stats["lep"]["tau_mag"])
        self.vis = stats["vis"]
        self.jet = stats["jet"]

    def _huber(self, value: torch.Tensor) -> torch.Tensor:
        absolute = value.abs()
        quadratic = torch.clamp(absolute, max=self.huber_delta)
        return 0.5 * quadratic**2 + self.huber_delta * (absolute - quadratic)

    def _cartesian(
        self, prediction: torch.Tensor, truth: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return self._huber((prediction - truth) / scale).sum(-1)

    def _phi(
        self,
        cos_hat: torch.Tensor,
        sin_hat: torch.Tensor,
        truth: torch.Tensor,
        tau_pt: float,
    ) -> torch.Tensor:
        px, py = truth[:, 0], truth[:, 1]
        pt = torch.sqrt(px * px + py * py + 1e-8)
        distance = 1.0 - (cos_hat * px / pt + sin_hat * py / pt)
        return pt / (pt + tau_pt) * distance

    def _latent(
        self,
        prediction: dict[str, torch.Tensor],
        truth: torch.Tensor,
        stats: dict[str, Any],
        *,
        clamp_pz: bool,
    ) -> torch.Tensor:
        px, py, pz = truth.unbind(-1)
        if clamp_pz:
            pz = pz.clamp_min(0.0)
        pt = torch.sqrt(px * px + py * py + 1e-8)
        zt = (
            torch.log1p(pt / float(stats["k_T"])) - float(stats["mu_uT"])
        ) / float(stats["sigma_uT"])
        zz = (
            torch.log1p(torch.clamp(pz / float(stats["k_Z"]), min=-0.999999))
            - float(stats["mu_uZ"])
        ) / float(stats["sigma_uZ"])
        return self._huber(prediction["latents"][:, 0] - zt) + self._huber(
            prediction["latents"][:, 1] - zz
        )

    def forward(
        self,
        pred_vis: dict[str, torch.Tensor],
        pred_jet: dict[str, torch.Tensor],
        true_vis: torch.Tensor,
        true_jet: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        vis_latent = self._latent(pred_vis, true_vis, self.vis, clamp_pz=False)
        jet_latent = self._latent(pred_jet, true_jet, self.jet, clamp_pz=True)
        vis_phi = self._phi(
            pred_vis["cos_phi"], pred_vis["sin_phi"], true_vis, self.tau_pt_vis
        )
        jet_phi = self._phi(
            pred_jet["cos_phi"], pred_jet["sin_phi"], true_jet, self.tau_pt_jet
        )
        vis_cart = self._cartesian(pred_vis["p_cart"], true_vis, self.s_vis_xyz)
        jet_cart = self._cartesian(pred_jet["p_cart"], true_jet, self.s_jet_xyz)

        true_lepton = true_vis - true_jet
        pred_lepton = pred_vis["p_cart"] - pred_jet["p_cart"]
        magnitude = true_lepton.norm(dim=-1)
        gate = magnitude / (magnitude + self.tau_mag_lep)
        lepton_consistency = self._cartesian(pred_lepton, true_lepton, self.s_lep_xyz)
        lepton_zero = self._cartesian(
            pred_lepton, torch.zeros_like(pred_lepton), self.s_lep_xyz
        )
        return {
            "loss_vis/latent": vis_latent,
            "loss_vis/phi": 0.5 * vis_phi,
            "loss_vis/cart": 0.15 * vis_cart,
            "loss_jet/latent": jet_latent,
            "loss_jet/phi": 0.5 * jet_phi,
            "loss_jet/cart": 0.15 * jet_cart,
            "loss_lep/cons": 0.5 * gate * lepton_consistency,
            "loss_lep/zero": 0.05 * (1.0 - gate) * lepton_zero,
        }


class MultiTaskObjective(nn.Module):
    TASKS = ("flavour", "charm", "vis_geom", "jet_geom", "lep_geom", "vertex")

    def __init__(
        self,
        metadata: dict[str, Any],
        *,
        label_smoothing: float,
        kendall_weight_min: float,
        kendall_weight_max: float,
    ):
        super().__init__()
        self.kinematics = CylindricalConsistencyLoss(metadata["target_stats"])
        self.label_smoothing = float(label_smoothing)
        self.weight_min = float(kendall_weight_min)
        self.weight_max = float(kendall_weight_max)
        initial = self._inverse_weight(1.0)
        self.uncertainty = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(initial)) for name in self.TASKS}
        )

    def _inverse_weight(self, weight: float) -> float:
        probability = (weight - self.weight_min) / (self.weight_max - self.weight_min)
        probability = min(max(probability, 1e-6), 1.0 - 1e-6)
        return math.log(probability / (1.0 - probability))

    def _weighted(
        self, name: str, loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.weight_min + (self.weight_max - self.weight_min) * torch.sigmoid(
            self.uncertainty[name]
        )
        regularizer = -0.5 * torch.log(weight.clamp_min(1e-12))
        return weight * loss + regularizer, weight

    def forward(
        self, outputs: dict[str, Any], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        smoothing = self.label_smoothing if self.training else 0.0
        flavour = F.cross_entropy(
            outputs["out_flavour"],
            targets["flavour_label"],
            reduction="none",
            label_smoothing=smoothing,
        )
        charm = F.cross_entropy(
            outputs["out_charm"],
            targets["charm_label"],
            reduction="none",
            label_smoothing=smoothing,
        )
        kin = self.kinematics(
            outputs["out_vis"],
            outputs["out_jet"],
            targets["vis_sp_momentum"],
            targets["jet_momentum"],
        )
        vis = kin["loss_vis/latent"] + kin["loss_vis/phi"] + kin["loss_vis/cart"]
        jet = kin["loss_jet/latent"] + kin["loss_jet/phi"] + kin["loss_jet/cart"]
        lepton = kin["loss_lep/cons"] + kin["loss_lep/zero"]
        vertex = F.mse_loss(
            outputs["out_vertex"], targets["primary_vertex"], reduction="none"
        ).mean(-1)

        losses = {
            "loss_cls/flavour": flavour,
            "loss_cls/charm": charm,
            "loss_vis/geom": vis,
            "loss_jet/geom": jet,
            "loss_lep/geom": lepton,
            "loss_vertex": vertex,
            **kin,
        }
        weighted = []
        weights = {}
        for task, value in (
            ("flavour", flavour),
            ("charm", charm),
            ("vis_geom", vis),
            ("jet_geom", jet),
            ("lep_geom", lepton),
            ("vertex", vertex),
        ):
            contribution, weight = self._weighted(task, value)
            weighted.append(contribution.mean())
            weights[f"uncertainty/{task}"] = weight
        total = torch.stack(weighted).sum()
        metrics = {"loss_total": total}
        metrics.update({name: value.mean() for name, value in losses.items()})
        metrics.update(weights)
        return total, metrics
