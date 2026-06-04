"""
GR + Massive-Graviton Frequency-Domain Waveform
================================================

GR baseline: IMRPhenomD via pycbc.waveform.get_fd_waveform
  (Khan et al. 2016, PRD 93 044007; arXiv:1508.07253)

Massive-graviton modification: Mirshekari, Yunes & Will 2012, PRD 85 024041
  (arXiv:1110.2720), ppE-formalism phase term

    delta_Psi_MG(f) = - pi^2 c D_L / [ lambda_g^2 (1+z) M_c f ]

  with:
    D_L     luminosity distance [m]
    lambda_g   graviton Compton wavelength [m]
    M_c     chirp mass in *seconds*  (M_c[s] = G M_c[kg] / c^3)
    f       GW frequency [Hz]

The modified waveform is:
    h_MG(f) = h_GR(f) * exp(i * delta_Psi_MG(f))

Numerical validation target (encoded in tests/test_waveform.py):
    For a GW150914-like injection at SNR=20, the match
    < h_GR | h_MG > / sqrt(<h_GR|h_GR><h_MG|h_MG>) drops below 0.99 when
    lambda_g ~ 1e16 km, in agreement with the LVK O3 90% upper limit
    lambda_g > 1.4e16 km (Abbott et al. 2021, PRD 103 122002).
"""
from __future__ import annotations

import numpy as np
from pycbc.waveform import get_fd_waveform

# Physics constants (SI)
G_N = 6.67430e-11          # m^3 kg^-1 s^-2
C_LIGHT = 2.99792458e8     # m/s
M_SUN = 1.98892e30         # kg
MPC_TO_M = 3.0857e22       # m / Mpc


def chirp_mass_kg(m1_msun: float, m2_msun: float) -> float:
    """M_c = (m1 m2)^(3/5) (m1+m2)^(-1/5), in kg."""
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def chirp_mass_seconds(m1_msun: float, m2_msun: float) -> float:
    """M_c in geometric units of seconds: M_c[s] = G M_c[kg] / c^3."""
    return G_N * chirp_mass_kg(m1_msun, m2_msun) / C_LIGHT ** 3


def dispersion_phase(
    freqs: np.ndarray,
    distance_mpc: float,
    chirp_mass_s: float,
    lambda_g_m: float,
    z: float = 0.0,
) -> np.ndarray:
    """
    Mirshekari-Yunes 2012 massive-graviton phase modification, eq. (2.2).

    delta_Psi(f) = - pi^2 c D_L / [ lambda_g^2 (1+z) M_c f ]

    For local sources (z << 1) this matches Will 1998 to leading order;
    D_0 -> D_L. For cosmological sources use the proper D_0 integral.
    """
    D_L = distance_mpc * MPC_TO_M
    safe_f = np.where(freqs > 0.0, freqs, np.inf)
    delta_psi = -(np.pi ** 2) * C_LIGHT * D_L / (
        lambda_g_m ** 2 * (1.0 + z) * chirp_mass_s * safe_f
    )
    return delta_psi


def gr_waveform_fd(
    m1_msun: float,
    m2_msun: float,
    distance_mpc: float,
    delta_f: float = 1.0 / 64.0,
    f_lower: float = 20.0,
    f_final: float = 2048.0,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
    inclination: float = 0.0,
    approximant: str = "IMRPhenomD",
) -> tuple[np.ndarray, np.ndarray]:
    """
    GR frequency-domain plus polarization h+(f) from pycbc.

    Returns (frequencies [Hz], h_plus [strain/Hz]).
    """
    hp, _ = get_fd_waveform(
        approximant=approximant,
        mass1=m1_msun, mass2=m2_msun,
        distance=distance_mpc,
        spin1z=spin1z, spin2z=spin2z,
        inclination=inclination,
        coa_phase=0.0,
        delta_f=delta_f, f_lower=f_lower, f_final=f_final,
    )
    freqs = hp.sample_frequencies.numpy().astype(np.float64)
    h_gr = hp.numpy().astype(np.complex128)
    return freqs, h_gr


def massive_graviton_waveform_fd(
    m1_msun: float,
    m2_msun: float,
    distance_mpc: float,
    lambda_g_m: float,
    z: float = 0.0,
    delta_f: float = 1.0 / 64.0,
    f_lower: float = 20.0,
    f_final: float = 2048.0,
    spin1z: float = 0.0,
    spin2z: float = 0.0,
    inclination: float = 0.0,
    approximant: str = "IMRPhenomD",
) -> tuple[np.ndarray, np.ndarray]:
    """
    h_MG(f) = h_GR(f) * exp(i * delta_Psi_MG(f))

    The GR baseline is IMRPhenomD; the dispersion phase is added on top
    of the IMR template. This is a leading-order treatment: it
    correctly captures dispersion in the inspiral but does not modify
    merger/ringdown morphology (a higher-order effect, see
    Calderon Bustillo et al. 2021).
    """
    freqs, h_gr = gr_waveform_fd(
        m1_msun, m2_msun, distance_mpc,
        delta_f=delta_f, f_lower=f_lower, f_final=f_final,
        spin1z=spin1z, spin2z=spin2z, inclination=inclination,
        approximant=approximant,
    )
    mc_s = chirp_mass_seconds(m1_msun, m2_msun)
    d_psi = dispersion_phase(freqs, distance_mpc, mc_s, lambda_g_m, z=z)
    h_mg = h_gr * np.exp(1j * d_psi)
    return freqs, h_mg
