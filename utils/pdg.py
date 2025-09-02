import numpy as np

# ----------------------------
# Label IDs
# ----------------------------
# 0 em_shower
# 1 muon_mip
# 2 z1_hadron_ion         (p/anti-p and Z=1 nuclei like d, t)
# 3 light_charged_hadron  (π±, K±)
# 4 neutral_hadron        (n, K0, Λ0/Σ0/Ξ0, neutral heavy mesons, …)
# 5 short_lived_hadron    (strange/charm/bottom with short cτ; charged or neutral)
# 6 ion_fragment          (Z>=2) or light_ion (Z=2..8) when split_ions=True
# 7 heavy_ion             (Z>=9) when split_ions=True (unused otherwise)
# 8 tau_or_ambiguous
LABEL_NAMES = {
    0: "em_shower",
    1: "muon_mip",
    2: "z1_hadron_ion",
    3: "light_charged_hadron",
    4: "neutral_hadron",
    5: "short_lived_hadron",
    6: "ion_fragment",
    7: "heavy_ion",
    8: "tau_or_ambiguous",
}

# ----------------------------
# Signature-oriented groups
# ----------------------------
EM = {22, 11, -11, 111, 221, 331}  # γ, e±, π0 (EM-like), 221 (η), 331 (η′).
MU = {13, -13}
PROTONS = {2212, -2212}
TAUS = {15, -15}

# Light charged hadrons: π±, K±
LIGHT_CHARGED = {211, -211, 321, -321}

# Neutral hadrons: n/nbar, K0 states, neutral hyperons, neutral heavy mesons
NEUTRAL_HADRONS = {
    2112, -2112,            # n, nbar
    130, 310,               # K_L, K_S
    311, -311,              # K0, K0bar (if present)
    3122, -3122,            # Λ0, anti-Λ0
    3212, -3212,            # Σ0, anti-Σ0
    3322, -3322,            # Ξ0, anti-Ξ0
    421, -421,              # D0, anti-D0
    511, -511,              # B0, anti-B0
    531, -531,              # B0_s, anti-B0_s
    4232, -4232,            # Ξ_c^0, anti (neutral)
}

# Short-lived strange/charm/bottom hadrons (charged or generic short-lived bucket)
SHORT_LIVED_HADRONS = {
    # strange baryons (charged)
    3112, -3112,            # Σ−, anti-Σ+
    3222, -3222,            # Σ+, anti-Σ−
    3312, -3312,            # Ξ−, anti-Ξ+
    3334, -3334,            # Ω−, anti-Ω+
    # charm (charged baryons/mesons)
    411, -411,              # D±
    431, -431,              # D_s±
    4122, -4122,            # Λ_c± (Λ_c+ & anti)
    # bottom (charged)
    521, -521,              # B±
    # (Bc±=541/-541 could be added here if present)
}

# ----------------------------
# Vectorised nuclear PDG decode
# ----------------------------
def _nuclear_Z_vectorized(pdg_abs: np.ndarray) -> np.ndarray:
    """
    Given |PDG| values, return Z for nuclear PDGs where |PDG|>=1e9.
    PDG nuclear code format: 1000000000 + 10LZZZAAAI.
    """
    code = pdg_abs - 1_000_000_000
    # Only valid for nuclei; mask others to zero later
    I = code % 10
    code //= 10
    A = code % 1000
    code //= 1000
    Z = code % 1000
    return Z

def cluster_labels_from_pdgs(pdgs: np.ndarray, split_ions: bool = False) -> np.ndarray:
    """
    Map an array of PDG codes -> array of group IDs (same shape).

    Parameters
    ----------
    pdgs : np.ndarray
        Array of ints (any shape) with PDG codes.
    split_ions : bool, default False
        If True: Z=2..8 -> 6 (light_ion), Z>=9 -> 7 (heavy_ion).
        If False: all Z>=2 -> 6 (ion_fragment).

    Returns
    -------
    np.ndarray
        Integer array of group IDs with the same shape as `pdgs`.
    """
    pdgs = np.asarray(pdgs)
    out = np.full(pdgs.shape, 5, dtype=np.int32)  # default to short_lived_hadron

    # Convenience
    abs_pdg = np.abs(pdgs)

    # 1) EM-like
    mask = np.isin(pdgs, list(EM))
    out[mask] = 0

    # 2) Muons
    mask = np.isin(pdgs, list(MU))
    out[mask] = 1

    # 3) Taus
    mask = np.isin(pdgs, list(TAUS))
    out[mask] = 8

    # 4) Nuclei / anti-nuclei (vectorized decode)
    is_nuc = abs_pdg >= 1_000_000_000
    if np.any(is_nuc):
        Z = _nuclear_Z_vectorized(abs_pdg[is_nuc])

        # Z == 1 -> z1_hadron_ion (like p/d/t stoppers)
        z1 = Z == 1
        idx = np.where(is_nuc)[0]
        out[idx[z1]] = 2

        # Z >= 2 -> ions (split or not)
        z_ge2 = Z >= 2
        if split_ions:
            light = (Z >= 2) & (Z <= 8)
            heavy = Z >= 9
            out[idx[light]] = 6  # light_ion
            out[idx[heavy]] = 7  # heavy_ion
        else:
            out[idx[z_ge2]] = 6  # ion_fragment

    # 5) Protons / antiprotons (non-nuclear p)
    mask = np.isin(pdgs, list(PROTONS))
    out[mask] = 2

    # 6) Light charged hadrons (π± / K±)
    mask = np.isin(pdgs, list(LIGHT_CHARGED))
    out[mask] = 3

    # 7) Neutral hadrons (n, K0, neutral hyperons, D0/B0/…)
    mask = np.isin(pdgs, list(NEUTRAL_HADRONS))
    out[mask] = 4

    # 8) Short-lived hadrons (charged strange/charm/bottom, etc.)
    mask = np.isin(pdgs, list(SHORT_LIVED_HADRONS))
    out[mask] = 5

    return out
