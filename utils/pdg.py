import numpy as np

# ----------------------------
# Contiguous label set (0..5)
# ----------------------------
# 0 em_shower
# 1 muon_mip
# 2 z1_hadron_ion      (p/p̄ and Z=1 nuclei: H, d, t, …)
# 3 charged_hadron     (π±, K±, τ hadronic, charged short-lived hadrons)
# 4 neutral_hadron     (n, K0 family, neutral hyperons, D0/B0/…)
# 5 ion_fragment       (all Z>=2 nuclei/anti-nuclei)
LABEL_NAMES = {
    0: "em_shower",
    1: "muon_mip",
    2: "hadron",
}

# Signature groups
EM = frozenset({22, 11, -11, 111, 221, 331})      # γ, e±, π0, η, η′
MU = frozenset({13, -13})
PROTONS = frozenset({2212, -2212})
LIGHT_CHARGED = frozenset({211, -211, 321, -321})
TAUS = frozenset({15, -15})

NEUTRAL_HADRONS = frozenset({
    2112, -2112,      # n, nbar
    130, 310,         # K_L, K_S
    311, -311,        # K0, K0bar
    3122, -3122,      # Λ0, anti-Λ0
    3212, -3212,      # Σ0, anti-Σ0
    3322, -3322,      # Ξ0, anti-Ξ0
    421, -421,        # D0, anti-D0
    511, -511,        # B0, anti-B0
    531, -531,        # B0_s, anti-B0_s
    4232, -4232,      # Ξ_c^0, anti
})

# Former "short_lived_hadron" (charged) → fold into charged_hadron
SHORT_LIVED_CHARGED = frozenset({
    3112, -3112, 3222, -3222, 3312, -3312, 3334, -3334,   # Σ±, Ξ−, Ω−
    411, -411, 431, -431, 4122, -4122, 521, -521,         # D±, D_s±, Λ_c±, B±
    # 541, -541  # B_c± if present
})

def _nuclear_Z_vectorized(pdg_abs: np.ndarray) -> np.ndarray:
    """Return Z for nuclear PDGs where |PDG|>=1e9 (1000000000 + 10LZZZAAAI)."""
    code = pdg_abs - 1_000_000_000
    _I = code % 10
    code //= 10
    _A = code % 1000
    code //= 1000
    Z = code % 1000
    return Z

def cluster_labels_from_pdgs(
    pdgs: np.ndarray,
    tau_decay_mode=-1,
    return_tau_flag: bool = False,
):
    """
    Map PDG codes → labels {0..5} with tau-mode routing.

    tau_decay_mode codes:
      1=e -> 0 (em_shower)
      2=mu -> 1 (muon_mip)
      3=1-prong -> 2 (charged_hadron)
      4=rho -> 2 (charged_hadron)
      5=3-prong -> 2 (charged_hadron)
      6=other -> 2 (charged_hadron)

    Notes:
      - Non-τ entries ignore tau_decay_mode.
      - All previously covered PDGs still map into {0..2}.
    """
    pdgs = np.asarray(pdgs)
    abs_pdg = np.abs(pdgs)

    # Default to charged-hadron signature
    out = np.full(pdgs.shape, 2, dtype=np.int32)

    # EM-like
    out[np.isin(pdgs, tuple(EM))] = 0

    # Muons
    out[np.isin(pdgs, tuple(MU))] = 1

    # Nuclei / anti-nuclei
    is_nuc = abs_pdg >= 1_000_000_000
    if np.any(is_nuc):
        Z = np.zeros_like(abs_pdg)
        Z[is_nuc] = _nuclear_Z_vectorized(abs_pdg[is_nuc])
        out[(Z == 1) & is_nuc] = 2          # Z == 1  -> z1_hadron_ion
        out[(Z >= 2) & is_nuc] = 2          # Z >= 2  -> ion_fragment

    # Non-nuclear protons / antiprotons
    out[np.isin(pdgs, tuple(PROTONS))] = 2

    # Light charged hadrons
    out[np.isin(pdgs, tuple(LIGHT_CHARGED))] = 2

    # Neutral hadrons
    out[np.isin(pdgs, tuple(NEUTRAL_HADRONS))] = 2

    # Charged short-lived hadrons -> charged_hadron
    out[np.isin(pdgs, tuple(SHORT_LIVED_CHARGED))] = 2

    # τ routing
    tau_mask = np.isin(pdgs, tuple(TAUS))
    if np.any(tau_mask):
        if np.any(tau_mask) and tau_decay_mode == -1:
            raise ValueError("tau_decay_mode must be provided when τ PDGs are present.")
        mode = int(tau_decay_mode)  # =1 e, =2 mu, =3 1-prong, =4 rho =5 3-prong, =6 other
        if mode == 1:
            out[tau_mask] = 0
        elif mode == 2:
            out[tau_mask] = 1
        else:
            out[tau_mask] = 2

    return (out, tau_mask) if return_tau_flag else out
