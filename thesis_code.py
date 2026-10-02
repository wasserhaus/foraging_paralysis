#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thesis_code.py
==============

Reproduction code for the Bachelor's thesis

    "Foraging Paralysis as a Collapse Mechanism --
     A Bifurcation Analysis of Honey Bee Colony Dynamics
     under Predation by Vespa velutina"
    Anton Fritzler, TU Muenchen, 2026

Every number the thesis states is recomputed here and compared with the value
printed in the thesis. Every numerical control is re-run with the protocol the
thesis states (integrator, tolerance, horizon). All figures of the thesis are
produced by the functions in Section 9.

The easiest way in is the notebook thesis_notebook.ipynb, which walks through
the thesis chapter by chapter and imports this file as a library.

Usage from a terminal
---------------------
    python thesis_code.py                 all checks + all figures (about 3 minutes)
    python thesis_code.py --quick         skip the finite-horizon threshold search (Tab. B.3)
    python thesis_code.py --no-figs       checks only
    python thesis_code.py --figs band,pm  only the named figures
    python thesis_code.py --sections cascade,stability
    python thesis_code.py --list          list sections and figures

Output
------
    verification_report.txt   every check: location, claim, thesis value, computed value, status
    verification_report.csv   the same, machine-readable
    figs/*.pdf, figs/*.png    the thesis figures (width = \\textwidth of the thesis, 16 cm)

Conventions
-----------
    State x = (H, F, f): hive bees, foragers, stores [g].
    Delay model y = (B, H, F, f) with the delay tau in the eclosion term.
    p = relative foraging activity, m = total forager mortality [1/d],
    V = hornet load (Requier et al. 2019 metric), p(V) = exp(-beta V).
    A check is PASS when the computed value agrees with the printed value to
    half a unit in the last printed digit, unless a wider tolerance is stated.

Requirements: Python >= 3.9, numpy, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize

# =============================================================================
# 0. PARAMETERS
# =============================================================================


@dataclass(frozen=True)
class Params:
    """Reference parameters (Notation table, Appendix A)."""
    L: float = 2000.0        # maximum laying rate [eggs/d]
    amin: float = 0.25       # minimum recruitment rate [1/d]
    amax: float = 0.25       # food-dependent recruitment amplitude [1/d]
    sigma: float = 0.75      # social inhibition [1/d]
    v: float = 5000.0        # half-saturation, brood care [bees]
    b: float = 500.0         # half-saturation, food [g]
    phi: float = 1.0 / 9.0   # pupation (eclosion) rate [1/d]
    tau: float = 12.0        # capped-brood delay [d]
    c: float = 0.10          # food collected per forager [g/d]
    gA: float = 0.007        # adult consumption [g/(bee d)]
    gB: float = 0.018        # brood consumption [g/(bee d)]
    m0: float = 0.154        # baseline forager mortality [1/d]
    # hornet calibration (Requier et al. 2019)
    beta: float = 0.109      # decay of foraging activity [1/hornet]
    h0: float = 0.0020       # homing failure at zero activity
    zeta: float = 6.09       # exponent of homing failure
    nu: float = 3.0          # foraging trips per day

    @property
    def kap(self) -> float:          # honey per bee reared [g]
        return self.gB / self.phi

    @property
    def omega(self) -> float:        # L / v
        return self.L / self.v

    @property
    def astar(self) -> float:        # largest admissible common weight [g]
        return self.kap + self.gA * self.v / self.L

    @property
    def asum(self) -> float:         # amin + amax
        return self.amin + self.amax


PAR = Params()

# initial states A-D (Appendix B.4) and the healthy state
STATES = {
    "A": (20000.0, 8000.0, 5000.0),
    "B": (9000.0, 2700.0, 1300.0),
    "C": (2000.0, 600.0, 600.0),
    "D": (200.0, 80.0, 400.0),
}
HEALTHY = STATES["B"]

# =============================================================================
# 1. REPORT
# =============================================================================


def _tol_from_str(s: str) -> float:
    """Half a unit in the last printed digit of a number given as a string."""
    s = s.strip().lower().replace("+", "")
    if "e" in s:
        mant, ex = s.split("e")
        d = len(mant.split(".")[1]) if "." in mant else 0
        return 0.5 * 10.0 ** (-d) * 10.0 ** int(ex)
    d = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10.0 ** (-d)


def _fmt(x) -> str:
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        ax = abs(float(x))
        if ax != 0 and (ax < 1e-3 or ax >= 1e6):
            return f"{float(x):.6e}"
        return f"{float(x):.8g}" if ax < 1e4 else f"{float(x):.2f}"
    return str(x)


class Report:
    """Collects every check and writes the verification report."""

    def __init__(self):
        self.rows = []
        self.section = ""

    def sec(self, name: str):
        self.section = name
        print(f"\n{'=' * 78}\n  {name}\n{'=' * 78}")

    def _add(self, status, where, what, claimed, computed, note=""):
        self.rows.append(dict(section=self.section, status=status, where=where,
                              what=what, thesis=str(claimed), code=_fmt(computed),
                              note=note))
        print(f"  [{status:4}] {where:<18} {what:<52} thesis={str(claimed):<12} "
              f"code={_fmt(computed)} {note}")

    def num(self, where, what, computed, claimed: str, tol: float | None = None,
            rel: float | None = None, note=""):
        """Compare a computed number with the value printed in the thesis."""
        cv = float(claimed)
        if rel is not None:
            ok = abs(computed - cv) <= rel * abs(cv)
        else:
            t = _tol_from_str(claimed) if tol is None else tol
            ok = abs(computed - cv) <= t * (1 + 1e-9) + 1e-15
        self._add("PASS" if ok else "DIFF", where, what, claimed, computed, note)
        return ok

    def true(self, where, what, cond: bool, computed="", note=""):
        """A qualitative claim (inequality, sign, ordering)."""
        self._add("PASS" if cond else "FAIL", where, what, "true", computed if computed != "" else cond, note)
        return cond

    def info(self, where, what, computed, note=""):
        """A value the thesis does not print but that documents a check."""
        self._add("INFO", where, what, "-", computed, note)

    def summary(self):
        n = len(self.rows)
        cnt = {k: sum(r["status"] == k for r in self.rows) for k in ("PASS", "DIFF", "FAIL", "INFO")}
        return n, cnt

    def write(self, path_txt: str, path_csv: str):
        n, cnt = self.summary()
        with open(path_txt, "w", encoding="utf-8") as fh:
            fh.write("Verification report -- Foraging Paralysis as a Collapse Mechanism\n")
            fh.write(time.strftime("generated %Y-%m-%d %H:%M\n\n"))
            fh.write(f"{n} checks: {cnt['PASS']} PASS, {cnt['DIFF']} DIFF, "
                     f"{cnt['FAIL']} FAIL, {cnt['INFO']} INFO\n")
            fh.write("PASS = agrees to the printed digits (or stated tolerance); DIFF = numerical "
                     "value differs; FAIL = qualitative claim not confirmed; INFO = documentation.\n")
            cur = None
            for r in self.rows:
                if r["section"] != cur:
                    cur = r["section"]
                    fh.write(f"\n--- {cur} ---\n")
                fh.write(f"[{r['status']:4}] {r['where']:<18} {r['what']:<52} "
                         f"thesis={r['thesis']:<12} code={r['code']} {r['note']}\n")
        with open(path_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


# =============================================================================
# 2. MODEL
# =============================================================================


def s_of(f, P=PAR):
    """Brood survival, food factor s(f) = f^2 / (f^2 + b^2)."""
    f2 = f * f
    return f2 / (f2 + P.b * P.b)


def sprime(f, P=PAR):
    return 2.0 * f * P.b ** 2 / (f * f + P.b ** 2) ** 2


def alpha_of(f, P=PAR):
    """Food-dependent recruitment alpha(f) = amin + amax b^2/(b^2+f^2)."""
    return P.amin + P.amax * (1.0 - s_of(f, P))


def rhs_M(t, x, p, m, P=PAR):
    """Right-hand side of model (M), Eq. (M)."""
    H, F, f = x
    N = H + F
    s = f * f / (f * f + P.b * P.b)
    S = s * H / (H + P.v)
    u = F / N if N > 0 else 0.0
    rec = H * (P.amin + P.amax * (1.0 - s) - P.sigma * u)
    return [P.L * S - rec, rec - m * F, P.c * p * F - P.gA * N - P.kap * P.L * S]


# ---- hornet functions (Section 4.3) -----------------------------------------

def p_of_V(V, P=PAR, beta=None):
    return np.exp(-(P.beta if beta is None else beta) * np.asarray(V, float))


def V_of_p(p, P=PAR, beta=None):
    return -np.log(np.asarray(p, float)) / (P.beta if beta is None else beta)


def HF_of_p(p, P=PAR):
    """Homing failure per return flight, HF(p) = h0 exp(-zeta p)."""
    return P.h0 * np.exp(-P.zeta * np.asarray(p, float))


def mu_of_p(p, P=PAR, k=1.0):
    """Additional forager mortality mu = nu p HF(p), Eq. (4.4); k scales HF."""
    return P.nu * np.asarray(p, float) * k * HF_of_p(p, P)


def mu_of_V(V, P=PAR, k=1.0):
    return mu_of_p(p_of_V(V, P), P, k)


def mu_bound(P=PAR):
    """Eq. (4.6): max_V mu = nu h0 / (zeta e)."""
    return P.nu * P.h0 / (P.zeta * math.e)


# =============================================================================
# 3. EQUILIBRIUM CASCADE AND THRESHOLDS (Sections 5.1, 5.5)
# =============================================================================


def J_Q(m, P=PAR):
    """Positive root of m J^2 + (m + sigma - amin) J - amin = 0, Eq. (5.5)."""
    bb = m + P.sigma - P.amin
    return (-bb + math.sqrt(bb * bb + 4.0 * m * P.amin)) / (2.0 * m)


def Q_of(m, P=PAR):
    return 1.0 / J_Q(m, P)


def P_ratio(p, m, P=PAR):
    """Food balance, Eq. (5.2): P = c p / gA - 1 - gB m / (gA phi)."""
    return P.c * p / P.gA - 1.0 - P.kap * m / P.gA


def bracket_of(J, m, P=PAR):
    """m J + sigma J/(1+J) - amin, the right-hand side of step (ii)."""
    return m * J + P.sigma * J / (1.0 + J) - P.amin


def equilibrium(p, m, P=PAR):
    """Positive equilibrium by the cascade (Lemma 5.3); None if it does not exist."""
    Pr = P_ratio(p, m, P)
    if Pr <= 0:
        return None
    J = 1.0 / Pr
    br = bracket_of(J, m, P)
    if not (0.0 < br < P.amax):
        return None
    psi = br / P.amax
    f = P.b * math.sqrt(1.0 / psi - 1.0)
    s = 1.0 - psi
    H = P.L * s / (m * J) - P.v
    if H <= 0:
        return None
    return np.array([H, J * H, f])


def s_star(m, P=PAR):
    """Positive root of Eq. (5.8)."""
    w = P.omega
    a2 = w * w + w * P.amax
    a1 = w * (P.sigma + m - P.asum) + m * P.amax
    a0 = -m * P.asum
    return (-a1 + math.sqrt(a1 * a1 - 4 * a2 * a0)) / (2 * a2)


def f_crit(m, P=PAR):
    s = s_star(m, P)
    return P.b * math.sqrt(s / (1.0 - s))


def P_c(m, P=PAR):
    return m * P.v / (P.L * s_star(m, P))


def p_X(X, m, P=PAR):
    """Common bookkeeping form, Eq. (5.30)."""
    return (P.gA * (1.0 + X) + P.kap * m) / P.c


def p_star(m, P=PAR):
    """Existence edge in closed form, Eq. (5.10)."""
    return p_X(P_c(m, P), m, P)


def p_acc(m, P=PAR):
    """Accumulation edge, Eq. (5.6)."""
    return p_X(Q_of(m, P), m, P)


def p_E(m, P=PAR, a=None):
    """Collapse bound p_E(a) = (gA + a m)/c, default a = a_star."""
    a = P.astar if a is None else a
    return (P.gA + a * m) / P.c


def p_kap(m, P=PAR):
    return (P.gA + P.kap * m) / P.c


def a_c(m, P=PAR):
    return P.kap + P.gA * P.v / (P.L * s_star(m, P))


def m_star(P=PAR):
    """Existence edge on the mortality axis at p = 1 (Corollary 5.8)."""
    return brentq(lambda m: p_star(m, P) - 1.0, 1e-3, 5.0, xtol=1e-14)


def m_acc_boundary(P=PAR):
    """Food-limitation boundary on the mortality axis at p = 1."""
    return brentq(lambda m: p_acc(m, P) - 1.0, 1e-3, 5.0, xtol=1e-14)


def threshold_with_m_of_p(which: str, m_of_p, P=PAR, lo=1e-3, hi=1.0):
    """Fixed point p = T(m(p)) for T in {acc, star, E}; used when m depends on p."""
    T = {"acc": p_acc, "star": p_star, "E": p_E}[which]
    return brentq(lambda p: p - T(m_of_p(p), P), lo, hi, xtol=1e-13)


# =============================================================================
# 4. JACOBIAN AND STABILITY (Section 5.2)
# =============================================================================


def jac_analytic(x, p, m, P=PAR):
    """Jacobian of (M), Eq. (5.12)."""
    H, F, f = x
    N = H + F
    s = s_of(f, P)
    g = H / (H + P.v)
    sp = sprime(f, P)
    R = P.amin + P.amax * (1 - s) - P.sigma * F / N
    Lsv = P.L * s * P.v / (H + P.v) ** 2
    sHF = P.sigma * H * F / N ** 2
    sHH = P.sigma * H * H / N ** 2
    return np.array([
        [Lsv - R - sHF, sHH, sp * (P.L * g + H * P.amax)],
        [R + sHF, -sHH - m, -H * P.amax * sp],
        [-P.gA - P.kap * Lsv, P.c * p - P.gA, -P.kap * P.L * g * sp],
    ])


def jac_reduced(x, p, m, P=PAR):
    """Jacobian after inserting the equilibrium relations, Eq. (5.14)."""
    H, F, f = x
    J = F / H
    g = H / (H + P.v)
    s = s_of(f, P)
    sp = sprime(f, P)
    q = P.sigma / (1 + J) ** 2
    return np.array([
        [-m * J * g - q * J, q, H * sp * (m * J / s + P.amax)],
        [m * J + q * J, -m - q, -P.amax * H * sp],
        [-P.gA - P.kap * m * J * (1 - g), P.kap * m + P.gA / J, -P.kap * m * J * H * sp / s],
    ])


def jac_numeric(fun, x, h=1e-6):
    x = np.asarray(x, float)
    n = len(x)
    Jm = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n)
        d[i] = h * max(1.0, abs(x[i]))
        Jm[:, i] = (np.asarray(fun(x + d)) - np.asarray(fun(x - d))) / (2 * d[i])
    return Jm


def char_coeffs(Jm):
    """a1, a2, a3 of lambda^3 + a1 lambda^2 + a2 lambda + a3."""
    a1 = -np.trace(Jm)
    a2 = (Jm[0, 0] * Jm[1, 1] - Jm[0, 1] * Jm[1, 0] + Jm[0, 0] * Jm[2, 2] - Jm[0, 2] * Jm[2, 0]
          + Jm[1, 1] * Jm[2, 2] - Jm[1, 2] * Jm[2, 1])
    a3 = -np.linalg.det(Jm)
    return a1, a2, a3


def a1_formula(x, m, P=PAR):
    """Reduced trace, Eq. (5.13)."""
    H, F, f = x
    J = F / H
    psi = 1 - s_of(f, P)
    return m * J * H / (H + P.v) + P.sigma / (1 + J) + m + 2 * P.kap * m * J * H * psi / f


def a3_formula(x, m, P=PAR):
    """Determinant, Eq. (5.15)."""
    H, F, f = x
    return m * P.gA * P.amax * H / (H + P.v) * H * sprime(f, P)


def slow_eig(p, m, P=PAR):
    x = equilibrium(p, m, P)
    ev = np.linalg.eigvals(jac_analytic(x, p, m, P))
    return ev[np.argmax(ev.real)], ev


# =============================================================================
# 5. FOOD AXIS (Section 5.4)
# =============================================================================


def A_of(f, P=PAR):
    return P.omega * s_of(f, P)


def u_star(f, m, P=PAR):
    """Interior fixed point of the shape equation near the axis, Eq. (5.27)."""
    return alpha_of(f, P) / (P.sigma + A_of(f, P) + m)


def lam_of(f, m, P=PAR):
    """Transverse growth rate lambda(f), Eq. (5.28)."""
    us = u_star(f, m, P)
    return A_of(f, P) * (1 - us) - m * us


def band_limits(m, P=PAR):
    """u_-, u^+ of Lemma 5.23."""
    return P.amin / (P.sigma + P.omega + m), P.asum / (P.sigma + m)


# =============================================================================
# 6. SIMULATION OF (M): LSODA WITH EXIT EVENT AND OPTIONAL RATIONING
# =============================================================================


def simulate_M(p, m, x0, T, rtol=1e-10, atol=1e-12, ration=False, stopN=None,
               P=PAR, max_step=np.inf, dense=True):
    """
    Integrate (M) with LSODA.

    - The face {f = 0} is monitored. Without rationing the run stops there
      (the solution leaves the model's domain Omega). With rationing
      (Proposition 5.19) the colony eats exactly its intake while
      f = 0 and c p F - gA (H + F) < 0.
    - stopN: terminal event H + F = stopN (e.g. 1 bee).

    Returns dict with t, x (3 x n), exited (t, N) or None, t_empty (time on the
    empty comb), stopped (time at stopN) or None.
    """
    ts, xs = [], []
    t0 = 0.0
    x = np.array(x0, float)
    exited = None
    t_empty = 0.0
    stopped = None

    def ev_f(t, y, *a):
        return y[2]
    ev_f.terminal = True
    ev_f.direction = -1

    def ev_N(t, y, *a):
        return y[0] + y[1] - stopN
    ev_N.terminal = True
    ev_N.direction = -1

    def ev_intake(t, y, *a):  # rationed phase ends when intake covers adult upkeep
        return P.c * p * y[1] - P.gA * (y[0] + y[1])
    ev_intake.terminal = True
    ev_intake.direction = 1

    def ev_N2(t, y, *a):
        return y[0] + y[1] - stopN
    ev_N2.terminal = True
    ev_N2.direction = -1

    def rhs_ration(t, y):
        H, F = y
        N = H + F
        u = F / N if N > 0 else 0.0
        rec = H * (P.amin + P.amax - P.sigma * u)   # alpha(0) = amin + amax, S = 0
        return [-rec, rec - m * F]

    while t0 < T:
        events = [ev_f] + ([ev_N] if stopN is not None else [])
        sol = solve_ivp(rhs_M, (t0, T), x, method="LSODA", args=(p, m, P), rtol=rtol,
                        atol=atol, events=events, max_step=max_step, dense_output=False)
        ts.append(sol.t)
        xs.append(sol.y)
        if sol.status == 1 and stopN is not None and len(sol.t_events[1]) > 0:
            stopped = sol.t_events[1][0]
            break
        if sol.status == 1 and len(sol.t_events[0]) > 0:
            te = sol.t_events[0][0]
            ye = sol.y_events[0][0].copy()
            ye[2] = 0.0
            if exited is None:
                exited = (te, ye[0] + ye[1])
            if not ration:
                break
            # rationed phase on the empty comb
            evs = [ev_intake] + ([ev_N2] if stopN is not None else [])
            sol2 = solve_ivp(rhs_ration, (te, T), ye[:2], method="LSODA", rtol=rtol, atol=atol,
                             events=evs)
            ts.append(sol2.t)
            xs.append(np.vstack([sol2.y, np.zeros_like(sol2.t)]))
            t_end = sol2.t[-1]
            t_empty += t_end - te
            if sol2.status == 1 and stopN is not None and len(sol2.t_events[1]) > 0:
                stopped = sol2.t_events[1][0]
                break
            if sol2.status == 1 and len(sol2.t_events[0]) > 0:
                t0 = t_end
                x = np.array([sol2.y[0, -1], sol2.y[1, -1], 1e-9])
                continue
            break
        break
    t = np.concatenate(ts)
    X = np.concatenate(xs, axis=1)
    return dict(t=t, x=X, exited=exited, t_empty=t_empty, stopped=stopped)


def W_of(x, a, P=PAR):
    """Energy capital W_a = f + a (H + F)."""
    return x[2] + a * (x[0] + x[1])


# =============================================================================
# 7. DELAY MODEL (K) WITH THE HORNET SUBSTITUTIONS: FIXED-STEP RK4
# =============================================================================


def dde_rk4(p, m, y0, T, dt=0.02, tau=None, B_hist=None, P=PAR, record=(),
            track_W=False, stop_all_below=None):
    """
    Fixed-step RK4 for the published four-dimensional delay model (K) with the
    substitutions c -> c p, m0 -> m, on a stored history of B (Appendix B.2).
    Works on floats (one colony) or numpy arrays (lanes of colonies with
    different p, m). B(t - tau + dt/2) is the mean of the two neighbouring grid
    values. tau = 0 gives the four-dimensional ODE with B as a state.

    record: iterable of times at which the state is stored.
    track_W: also return max increase of W_D = f + a*(H+F+Pi) + (a*-kap) B.
    Returns dict(rec={time: (B,H,F,f)}, final=(B,H,F,f), dWmax=..., Wtrace=...).
    """
    L, amin, amax, sig, v = P.L, P.amin, P.amax, P.sigma, P.v
    b2, phi, c, gA, gB = P.b * P.b, P.phi, P.c, P.gA, P.gB
    tau = P.tau if tau is None else tau
    B, H, F, f = y0
    nd = int(round(tau / dt))
    if B_hist is None:
        B_hist = 0.0 * B
    hist = deque([B_hist] * nd + [B], maxlen=nd + 1) if nd > 0 else None
    cp = c * p

    def rhs(B, H, F, f, Bd):
        f2 = f * f
        s = f2 / (f2 + b2)
        N = H + F
        S = s * H / (H + v)
        rec = H * (amin + amax * (1.0 - s) - sig * F / (N + 1e-300))
        return (L * S - phi * B, phi * Bd - rec, rec - m * F, cp * F - gA * N - gB * B)

    nsteps = int(round(T / dt))
    rec_steps = {int(round(tr / dt)): tr for tr in record}
    out = {}
    astar, kap = P.astar, P.kap
    runsum = sum(hist) if nd > 0 else 0.0
    dWmax = -np.inf
    Wtrace = []
    W_prev = None
    if track_W:
        Pi = phi * dt * (runsum - 0.5 * (hist[0] + hist[-1])) if nd > 0 else 0.0
        W_prev = f + astar * (H + F + Pi) + (astar - kap) * B
        Wtrace.append((0.0, W_prev))
    for k in range(1, nsteps + 1):
        if nd > 0:
            d0 = hist[0]
            d1 = hist[1]
            dm = 0.5 * (d0 + d1)
            k1 = rhs(B, H, F, f, d0)
            h2 = 0.5 * dt
            k2 = rhs(B + h2 * k1[0], H + h2 * k1[1], F + h2 * k1[2], f + h2 * k1[3], dm)
            k3 = rhs(B + h2 * k2[0], H + h2 * k2[1], F + h2 * k2[2], f + h2 * k2[3], dm)
            k4 = rhs(B + dt * k3[0], H + dt * k3[1], F + dt * k3[2], f + dt * k3[3], d1)
        else:
            k1 = rhs(B, H, F, f, B)
            h2 = 0.5 * dt
            B2 = B + h2 * k1[0]
            k2 = rhs(B2, H + h2 * k1[1], F + h2 * k1[2], f + h2 * k1[3], B2)
            B3 = B + h2 * k2[0]
            k3 = rhs(B3, H + h2 * k2[1], F + h2 * k2[2], f + h2 * k2[3], B3)
            B4 = B + dt * k3[0]
            k4 = rhs(B4, H + dt * k3[1], F + dt * k3[2], f + dt * k3[3], B4)
        w = dt / 6.0
        B = B + w * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        H = H + w * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        F = F + w * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        f = f + w * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        if nd > 0:
            runsum = runsum - hist[0] + B
            hist.append(B)
        if track_W:
            Pi = phi * dt * (runsum - 0.5 * (hist[0] + hist[-1])) if nd > 0 else 0.0
            W = f + astar * (H + F + Pi) + (astar - kap) * B
            dWmax = max(dWmax, float(np.max(W - W_prev)))
            W_prev = W
            if k % 50 == 0:
                Wtrace.append((k * dt, W))
        if k in rec_steps:
            out[rec_steps[k]] = (B, H, F, f)
        if stop_all_below is not None and k % 500 == 0:
            if np.all(H + F < stop_all_below):
                break
    return dict(rec=out, final=(B, H, F, f), dWmax=dWmax, Wtrace=Wtrace, t_end=k * dt)


def delay_linearisation(p, m, P=PAR):
    """A0, A1 of the linearised delay model at the common equilibrium."""
    x = equilibrium(p, m, P)
    H, F, f = x
    Bst = m * F / P.phi

    def G(y, Bd):
        B, H, F, f = y
        s = s_of(f, P)
        N = H + F
        S = s * H / (H + P.v)
        rec = H * (P.amin + P.amax * (1 - s) - P.sigma * F / N)
        return np.array([P.L * S - P.phi * B, P.phi * Bd - rec, rec - m * F,
                         P.c * p * F - P.gA * N - P.gB * B])

    y = np.array([Bst, H, F, f])
    A0 = jac_numeric(lambda z: G(z, Bst), y, h=1e-7)
    A1 = np.zeros((4, 4))
    A1[1, 0] = P.phi
    return A0, A1, y


def char_D(lam, A0, A1, tau):
    lam = np.atleast_1d(lam)
    I = np.eye(4)
    M = lam[:, None, None] * I - A0[None] - A1[None] * np.exp(-lam * tau)[:, None, None]
    return np.linalg.det(M)


def rightmost_real_root(A0, A1, tau, guess):
    """Real root of D near the slow eigenvalue of (M) (Newton / bracketing)."""
    g = lambda l: char_D(l, A0, A1, tau)[0].real
    lo, hi = guess * 3.0, guess * 0.2
    # expand bracket if needed
    for _ in range(60):
        if g(lo) * g(hi) < 0:
            break
        lo *= 1.5
        hi *= 0.5
    return brentq(g, lo, hi, xtol=1e-15)


def count_roots(A0, A1, tau, sigma0, R=30.0, n=160000):
    """Number of roots of D in {Re >= sigma0, |lam - sigma0| <= R} (argument principle)."""
    n1 = n // 2
    # vertical segment sigma0 + iR -> sigma0 - iR, dense near the real axis
    t = np.linspace(-1, 1, n1)
    y = R * np.sign(t) * np.abs(t) ** 3
    seg = sigma0 + 1j * y[::-1]
    th = np.linspace(-np.pi / 2, np.pi / 2, n - n1)
    arc = sigma0 + R * np.exp(1j * th)
    z = np.concatenate([seg, arc, seg[:1]])
    D = char_D(z, A0, A1, tau)
    ang = np.unwrap(np.angle(D))
    return int(round((ang[-1] - ang[0]) / (2 * np.pi)))


# =============================================================================
# 8. SECTIONS OF CHECKS
# =============================================================================

def sec_parameters(rep: Report, P=PAR, full=False):
    rep.sec("Parameters, calibration and hornet functions (Ch. 3-4, App. A)")
    rep.num("Notation", "kappa = gB/phi [g]", P.kap, "0.1620")
    rep.num("Sec. 3.2.2", "gB = 0.163 g / 9 d", 0.163 / 9, "0.018", tol=2e-4)
    rep.num("Eq. (5.19)", "a_star = kappa + gA v / L [g]", P.astar, "0.1795")
    rep.num("Rem. 5.15", "gA v / L [g]", P.gA * P.v / P.L, "0.0175")
    rep.num("Rem. 5.15", "L / v, bees reared per nurse-day", P.omega, "0.4")
    rep.num("Rem. 5.15", "m0 v / L, nurses per forager", P.m0 * P.v / P.L, "0.385")
    rep.num("Sec. 3.1.1", "1/m0 [d]", 1 / P.m0, "6.49")
    rep.num("Sec. 3.3", "c / gA, forager earns x her keep", P.c / P.gA, "14", tol=0.5)
    rep.num("Sec. 3.3", "kappa m0 [g/d]", P.kap * P.m0, "0.025")
    rep.num("App. A", "k = gB/(gA phi)", P.gB / (P.gA * P.phi), "23.1429")

    # Khoury 2011 critical death rate: (1-u)(alpha - sigma u) = L/w at m u = L/w
    w = 27000.0
    Lw = P.L / w
    # solve (1-u)(amin - sigma u) = L/w for the root in (0, amin/sigma)
    uc = brentq(lambda uu: (1 - uu) * (P.amin - P.sigma * uu) - Lw, 0.0, P.amin / P.sigma)
    mc = Lw / uc
    rep.num("Sec. 3.3", "critical death rate of the 2011 model", mc, "0.355")
    rep.num("Sec. 3.3", "critical flightspan 1/0.355 [d]", 1 / 0.355, "2.8")
    rep.num("Sec. 3.1", "reversion above forager fraction amin/sigma", P.amin / P.sigma, "0.3333", tol=1e-4)

    # digitisation of Requier Fig. 2a
    Vd = np.array([5.0, 10.0, 15.0, 20.0])
    pd = np.array([0.55, 0.33, 0.20, 0.11])
    lnp = -np.log(pd)
    num = float(np.sum(Vd * lnp))
    beta_ls = num / float(np.sum(Vd ** 2))
    rep.num("Eq. (B.1)", "numerator sum V_i(-ln p_i)", num, "82.363")
    rep.num("Eq. (B.1)", "least-squares beta through origin", beta_ls, "0.1098")
    bi = lnp / Vd
    for Vi, bb, cl in zip(Vd, bi, ["0.1196", "0.1109", "0.1073", "0.1104"]):
        rep.num("Tab. B.1", f"pointwise beta at V={Vi:.0f}", bb, cl)
    rep.num("Sec. 4.3.2", "pointwise spread min", bi.min(), "0.107")
    rep.num("Sec. 4.3.2", "pointwise spread max", bi.max(), "0.120")
    for Vi, cl in zip(Vd, ["0.577", "0.334", "0.193", "0.111"]):
        rep.num("Tab. B.1", f"fit exp(-0.1098 V) at V={Vi:.0f}", math.exp(-0.1098 * Vi), cl,
                note="(0.5775 -> 0.578)" if Vi == 5 else "")

    # digitisation of Requier Fig. 2b
    ph = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    hf = np.array([0.202, 0.110, 0.060, 0.032, 0.018, 0.010]) / 100
    zi = -np.log(hf[1:] / hf[0]) / ph[1:]
    for pi_, z, cl in zip(ph[1:], zi, ["6.078", "6.070", "6.142", "6.045", "6.011"]):
        rep.num("Tab. B.2", f"pointwise zeta at p={pi_:.1f}", z, cl)
    slope, icpt = np.polyfit(ph, np.log(hf), 1)
    rep.num("App. B.1", "LS fit of ln HF: zeta", -slope, "6.03",
            note="(6.02496: rounds to 6.02, write 6.025)")
    rep.num("App. B.1", "LS fit of ln HF: prefactor [%]", 100 * math.exp(icpt), "0.200")
    for pi_, cl in zip(ph, ["0.200", "0.110", "0.060", "0.033", "0.018", "0.010"]):
        rep.num("Tab. B.2", f"fitted curve (unrounded) at p={pi_:.1f} [%]",
                100 * math.exp(icpt + slope * pi_), cl)
    rep.true("Sec. 4.3.3", "zeta = 6.09 inside pointwise interval [6.01,6.14]", 6.01 <= P.zeta <= 6.14)
    # logistic vs exponential for probabilities below 0.2 %
    q = np.linspace(1e-9, 0.002, 1000)
    # exp(eta)/logistic(eta) - 1 = odds = q/(1-q) ~ q
    rep.true("Sec. 4.3.3", "logit vs exponential differ by odds q/(1-q) <= 0.2004 %",
             float(np.max(q / (1 - q))) <= 0.0020041, float(np.max(q / (1 - q))),
             note="(exactly 0.2004 % at HF = 0.2 %: 'to within 0.2 %' is the safe wording)")
    rep.num("Sec. 4.3.3", "HF(0) [%]", 100 * float(HF_of_p(0.0)), "0.20")
    rep.num("Sec. 4.3.3", "HF(1)", float(HF_of_p(1.0)), "4.5e-6")
    rep.info("Sec. 4.3.3", "orders of magnitude HF(0)/HF(1)", math.log10(float(HF_of_p(0) / HF_of_p(1))),
             "(thesis: between two and three)")
    rep.num("Sec. 1.2", "HF(0) = one flight in", 1 / float(HF_of_p(0.0)), "500")
    rep.true("Sec. 1.2", "traffic falls by roughly an order of magnitude over 0-20 hornets",
             5 < 1 / float(p_of_V(20.0)) < 20, 1 / float(p_of_V(20.0)))

    # trips per day and closure
    rep.num("Eq. (4.3)", "load per trip c/nu [mg]", 1000 * P.c / P.nu, "33.3")
    rep.num("Eq. (4.5)", "17 trips / nu [d]", 17 / P.nu, "5.7")
    rep.num("Eq. (4.5)", "21 trips / nu [d]", 21 / P.nu, "7.0")
    rep.true("Eq. (4.5)", "1/m0 inside [5.7, 7.0]", 17 / P.nu <= 1 / P.m0 <= 21 / P.nu)

    # the mortality bound
    mb = mu_bound(P)
    rep.num("Eq. (4.6)", "max mu = nu h0/(zeta e) [1/d]", mb, "3.62e-4")
    rep.num("Eq. (4.6)", "max mu / m0 [%]", 100 * mb / P.m0, "0.235")
    rep.num("Eq. (4.6)", "argmax p = 1/zeta", 1 / P.zeta, "0.164")
    rep.num("Eq. (4.6)", "argmax V", float(V_of_p(1 / P.zeta)), "16.6")
    Vg = np.linspace(0, 60, 600001)
    rep.true("Eq. (4.6)", "mu(V) <= bound on V in [0,60]", float(np.max(mu_of_V(Vg))) <= mb * (1 + 1e-12))
    mb603 = mu_bound(replace(P, zeta=6.03))
    rep.num("Sec. 4.3.3", "zeta=6.03 raises bound by [%]", 100 * (mb603 / mb - 1), "1", tol=0.1)
    # thresholds with mu carried, zeta 6.09 vs 6.03
    shifts = []
    for which in ("acc", "star", "E"):
        pa = threshold_with_m_of_p(which, lambda p: P.m0 + mu_of_p(p, P), P)
        P2 = replace(P, zeta=6.03)
        pb = threshold_with_m_of_p(which, lambda p: P.m0 + mu_of_p(p, P2), P)
        shifts.append(abs(float(V_of_p(pa)) - float(V_of_p(pb))))
    rep.true("Sec. 4.3.3", "zeta 6.03 moves no threshold by 0.001 hornets", max(shifts) < 1e-3, max(shifts))

    # Chapter 4 closing numbers
    rep.num("Sec. 4.6", "p(5): traffic after five hornets", float(p_of_V(5)), "0.58")
    rep.num("Sec. 4.6", "p(10): traffic after ten hornets", float(p_of_V(10)), "0.34", tol=0.01,
            note="thesis: 'to a third'")

    # trip-rate table (Tab. A.4)
    for nu, cl in [(3.0, "0.24"), (2.91, "0.23"), (24.0, "1.88")]:
        rep.num("Tab. A.4", f"max mu/m0 [%] for nu={nu}", 100 * mu_bound(replace(P, nu=nu)) / P.m0, cl)


def sec_khoury(rep: Report, P=PAR, full=False):
    rep.sec("Reproduction of Khoury et al. 2013 (App. C)")
    m = 0.42
    Pr = P_ratio(1.0, m, P)
    rep.num("App. C.1.2", "P at m=0.42", Pr, "3.5657")
    br = m / Pr + P.sigma / (1 + Pr) - P.amin
    rep.num("App. C.1.2", "denominator m/P + sigma/(1+P) - amin", br, "0.03206")
    X = br / P.amax
    rep.num("App. C.1.2", "X", X, "0.12824", note="(thesis divides the rounded 0.03206)")
    x = equilibrium(1.0, m, P)
    rep.num("App. C.1.2", "f* [g]", x[2], "1303.7")
    rep.num("App. C.1.2", "s(f*)", s_of(x[2]), "0.87175", note="(exact 0.87178)")
    rep.num("App. C.1.2", "H*", x[0], "9802.4")
    rep.num("App. C.1.2", "F*", x[1], "2749.1")
    rep.num("App. C.1.2", "B* = m F/phi", m * x[1] / P.phi, "10391.6", note="(thesis uses rounded F=2749.1)")
    printed = P.sigma / (Pr + 1) - m / Pr - P.amin
    rep.num("App. C.1.3(a)", "printed denominator (sign error)", printed, "-0.20352")
    rep.true("App. C.1.3(a)", "printed formula has no real f", P.amax / printed - 1 < 0)
    k = P.gB / (P.gA * P.phi)
    A = P.c / P.gA - 1
    rep.num("App. C.1.3(b)", "A = c/gA - 1", A, "13.2857")
    co = [k * (P.amin * k + 1), -(2 * P.amin * A * k + (P.amin - P.sigma) * k + A + 1),
          A * (P.amin * A + P.amin - P.sigma)]
    rep.num("App. C.1.3(b)", "correct m^2 coefficient", co[0], "157.04")
    r = np.sort(np.roots(co).real)
    rep.num("App. C.1.3(b)", "true quadratic, root 1", r[0], "0.4010")
    rep.num("App. C.1.3(b)", "true quadratic, root 2", r[1], "0.5952")
    pub = [k, -(P.sigma * k - P.c / P.gA), P.asum + P.sigma - P.sigma * P.c / P.gA]
    rp = np.sort(np.roots(pub).real)
    rep.num("App. C.1.3(b)", "published quadratic, root 1", rp[0], "-0.5766")
    rep.num("App. C.1.3(b)", "published quadratic, root 2", rp[1], "0.7093")
    A8 = P.phi * P.gA / P.gB * (P.c / P.gA - 1)
    rep.num("Eq. (C.7)", "(A8) = 31/54", A8, "0.5741")
    rep.true("Eq. (C.7)", "(A8) equals 31/54 exactly", abs(A8 - 31 / 54) < 1e-14)
    # order in which the conditions bind (Tab. C.1)
    m_PQ = m_acc_boundary(P)
    m_H0 = m_star(P)
    m_den = brentq(lambda mm: mm / P_ratio(1, mm, P) + P.sigma / (1 + P_ratio(1, mm, P)) - P.amin - P.amax,
                   0.42, 0.57)
    m_P0 = brentq(lambda mm: P_ratio(1, mm, P), 0.3, 1.0)
    rep.num("Tab. C.1", "P = Q", m_PQ, "0.4010")
    rep.num("Tab. C.1", "H* = 0", m_H0, "0.4624")
    rep.num("Tab. C.1", "denominator = amax", m_den, "0.4888")
    rep.num("Tab. C.1", "P = 0", m_P0, "0.5741")
    rep.true("Tab. C.1", "binding order 0.4010 < 0.4624 < 0.4888 < 0.5741", m_PQ < m_H0 < m_den < m_P0)
    rep.num("Eq. (C.8)", "f*(m*) [g]", f_crit(m_H0, P), "449.8")
    # food-abundant regime (A4)
    Q = Q_of(P.m0, P)
    Fa = (P.L * Q - P.m0 * P.v) / (P.m0 * Q)
    rep.info("Eq. (C.2)", "food-abundant F at m0 (A4)", Fa)
    if full:
        # direct integration of the delay model at m = 0.42, p = 1
        t0 = time.time()
        res = dde_rk4(1.0, 0.42, (0.0, 16000.0, 8000.0, 0.0), 12000.0, dt=0.02, P=P)
        B, H, F, f = res["final"]
        for nm, val, cl in [("B", B, 10391.6), ("H", H, 9802.4), ("F", F, 2749.1), ("f", f, 1303.7)]:
            rep.num("App. C.1.2", f"delay model integrated, {nm} (5 s.f.)", val, str(cl), rel=5e-5)
        rep.info("App. C.1.2", "runtime [s]", time.time() - t0)


def sec_cascade(rep: Report, P=PAR, full=False):
    rep.sec("Equilibrium cascade, existence band and thresholds (Sec. 5.1, 5.5)")
    m = P.m0
    JQ = J_Q(m, P)
    Q = 1 / JQ
    rep.num("Eq. (5.5)", "J_Q", JQ, "0.352932")
    rep.num("Eq. (5.5)", "Q", Q, "2.8334")
    Pp1 = P_ratio(1.0, m, P)
    rep.num("Sec. 5.1.1", "P at p=1", Pp1, "9.7217")
    rep.num("Sec. 5.1.1", "Q as share of P [%]", 100 * Q / Pp1, "29.1")
    pa = p_acc(m, P)
    rep.num("Eq. (5.6)", "p_acc", pa, "0.517818")
    rep.num("Eq. (5.6)", "V_acc", float(V_of_p(pa)), "6.04")
    rep.num("Eq. (5.6)", "food-limitation boundary in m at p=1", m_acc_boundary(P), "0.4010")
    ss = s_star(m, P)
    rep.num("Prop. 5.6", "s*", ss, "0.281699")
    rep.num("Prop. 5.6", "f_crit [g]", f_crit(m, P), "313.12")
    rep.num("Prop. 5.6", "J_c", 1 / P_c(m, P), "0.7317")
    rep.num("Prop. 5.6", "P_c", P_c(m, P), "1.3667")
    ps = p_star(m, P)
    rep.num("Eq. (5.10)", "p*", ps, "0.415150")
    rep.num("Eq. (5.10)", "V*", float(V_of_p(ps)), "8.07")
    ms = m_star(P)
    rep.num("Cor. 5.8", "m*", ms, "0.4624055")
    rep.num("Cor. 5.8", "m*/m0", ms / m, "3.00")
    rep.num("Cor. 5.8", "m* - m0", ms - m, "0.3084")
    rep.num("Prop. 5.6", "s*(m*)", s_star(ms, P), "0.447317")
    rep.num("Prop. 5.6", "f_crit(m*) [g]", f_crit(ms, P), "449.82")
    # the band by scanning
    pg = np.linspace(0.30, 0.60, 30001)
    ex = np.array([equilibrium(p, m, P) is not None for p in pg])
    inside = (pg > ps) & (pg < pa)
    rep.true("Prop. 5.5", "equilibrium exists iff p* < p < p_acc (scan of 30001 p)",
             bool(np.all(ex == inside)))
    # condition 2 upper bound: bracket = amax below p*
    p_up = brentq(lambda p: bracket_of(1 / P_ratio(p, m, P), m, P) - P.amax, P.kap * m / P.c + P.gA / P.c + 1e-6, ps)
    rep.true("Prop. 5.5", "upper bound of cond. 2 fails only below p*", p_up < ps, p_up)
    rep.true("Prop. 5.5", "condition 1 (P>0) holds on the band", P_ratio(ps, m, P) > 0)
    # collision: branch -> (0, 0, f_crit)
    xe = equilibrium(ps + 1e-9, m, P)
    rep.true("Prop. 5.27", "branch -> (0,0,f_crit) as p -> p*",
             xe[0] < 1e-3 and abs(xe[2] - f_crit(m, P)) < 1e-3, f"H={xe[0]:.2e}, f={xe[2]:.4f}")
    # dH*/dp >= L c s / (m gA)
    ok = True
    for p in np.linspace(ps + 1e-4, pa - 1e-4, 200):
        h = 1e-7
        dH = (equilibrium(p + h, m, P)[0] - equilibrium(p - h, m, P)[0]) / (2 * h)
        lb = P.L * P.c * s_of(equilibrium(p, m, P)[2]) / (m * P.gA)
        ok &= dH >= lb * (1 - 1e-6)
    rep.true("Sec. 5.4.2", "dH*/dp >= L c s(f*)/(m gA) on the band", ok)
    # monotonicity in m (App. D.6, Sec. 6.3)
    mg = np.linspace(0.02, ms, 400)
    rep.true("App. D.6", "p*(m) strictly increasing", bool(np.all(np.diff([p_star(mm, P) for mm in mg]) > 0)))
    rep.true("Sec. 6.3", "p_acc(m) strictly increasing", bool(np.all(np.diff([p_acc(mm, P) for mm in mg]) > 0)))
    rep.true("Sec. 6.3", "Q(m) increasing", bool(np.all(np.diff([Q_of(mm, P) for mm in mg]) > 0)))
    # f_crit independent of p, c, gA, gB
    Pv = replace(P, c=0.13, gA=0.009, gB=0.02)
    rep.true("Rem. 5.29", "f_crit independent of c, gA, gB", abs(f_crit(m, Pv) - f_crit(m, P)) < 1e-12)
    # ladder
    pk = p_kap(m, P)
    pe = p_E(m, P)
    rep.num("Eq. (5.16)", "p_kappa", pk, "0.319480")
    rep.num("Eq. (5.16)", "V at p_kappa", float(V_of_p(pk)), "10.47")
    rep.num("Eq. (5.22)", "p_E", pe, "0.346430")
    rep.num("Eq. (5.22)", "V_E", float(V_of_p(pe)), "9.73")
    rep.num("Tab. 5.2", "X for p_E = m v / L", m * P.v / P.L, "0.385")
    rep.num("Tab. 5.2", "X for p* = P_c", P_c(m, P), "1.367")
    rep.num("Tab. 5.2", "X for p_acc = Q", Q, "2.833")
    for X, pv in [(0, pk), (m * P.v / P.L, pe), (P_c(m, P), ps), (Q, pa)]:
        rep.true("Eq. (5.30)", f"p_X({X:.3f}) reproduces threshold", abs(p_X(X, m, P) - pv) < 1e-13)
    rep.true("Prop. 5.32", "order p_kappa < p_E < p* < p_acc", pk < pe < ps < pa)
    ww = P.gA * m * P.v / (P.c * P.L) * (1 - ss) / ss
    rep.num("Eq. (5.31)", "window width p* - p_E", ww, "0.0687")
    rep.true("Eq. (5.31)", "closed form equals p* - p_E", abs(ww - (ps - pe)) < 1e-13)
    rep.num("Eq. (5.31)", "window width in hornets", float(V_of_p(pe) - V_of_p(ps)), "1.66")
    rep.num("Prop. 5.33", "a_c [g]", a_c(m, P), "0.2241")
    rep.true("App. D.8", "p_E(a_c) = p*", abs(p_E(m, P, a=a_c(m, P)) - ps) < 1e-13)
    rep.num("Sec. 5.1.2", "band width in hornets", float(V_of_p(ps) - V_of_p(pa)), "2", tol=0.1,
            note="thesis: 'about two hornets wide'")
    rep.num("Sec. 6.1", "existence edge: activity falls by [%]", 100 * (1 - ps), "58")


def sec_random_parameters(rep: Report, P=PAR, full=False, n=20000, seed=1):
    """Claims stated 'for all positive parameters': tested on random parameter sets."""
    rep.sec("Claims for all parameters: random-parameter tests")
    rng = np.random.default_rng(seed)
    n_eq = n_ok_a1 = n_ok_a3 = n_ok_rh = n_ok_red = 0
    n_ord = n_ord_ok = n_lam = n_lam_ok = n_mono = n_mono_ok = 0
    worst_rh = np.inf
    for _ in range(n):
        f = lambda: math.exp(rng.uniform(math.log(0.3), math.log(3.0)))
        Q_ = replace(P, L=P.L * f(), amin=P.amin * f(), amax=P.amax * f(), sigma=P.sigma * f(),
                     v=P.v * f(), b=P.b * f(), phi=P.phi * f(), c=P.c * f(), gA=P.gA * f(),
                     gB=P.gB * f())
        m = P.m0 * f()
        p = rng.uniform(0.05, 1.0)
        x = equilibrium(p, m, Q_)
        if x is not None:
            n_eq += 1
            Ja = jac_analytic(x, p, m, Q_)
            Jn = jac_numeric(lambda z: rhs_M(0, z, p, m, Q_), x, h=1e-6)
            a1, a2, a3 = char_coeffs(Ja)
            n_ok_a1 += abs(a1 - a1_formula(x, m, Q_)) <= 1e-8 * max(1, abs(a1))
            n_ok_a3 += abs(a3 - a3_formula(x, m, Q_)) <= 1e-7 * max(1e-12, abs(a3))
            n_ok_red += np.allclose(Ja, jac_reduced(x, p, m, Q_), rtol=1e-8, atol=1e-10) and \
                np.allclose(Ja, Jn, rtol=1e-4, atol=1e-7)
            n_ok_rh += (a1 > 0 and a3 > 0 and a1 * a2 > a3)
            worst_rh = min(worst_rh, (a1 * a2 - a3) / (a1 * a2))
        # ordering and lambda monotonicity under sigma > amin + amax and L/v > amin
        if Q_.sigma > Q_.asum and Q_.omega > Q_.amin:
            n_ord += 1
            try:
                ok = p_kap(m, Q_) < p_E(m, Q_) < p_star(m, Q_) < p_acc(m, Q_)
                n_ord_ok += ok
            except (ValueError, ZeroDivisionError):
                pass
            fg = np.linspace(1e-3, 20 * Q_.b, 400)
            lv = lam_of(fg, m, Q_)
            n_lam += 1
            n_lam_ok += bool(np.all(np.diff(lv) > 0)) and abs(lam_of(f_crit(m, Q_), m, Q_)) < 1e-9
        if Q_.sigma > Q_.asum:
            n_mono += 1
            mg = np.linspace(0.05, 2.0, 60) * m
            n_mono_ok += bool(np.all(np.diff([p_star(mm, Q_) for mm in mg]) > 0))
    rep.info("Thm. 5.10", "random parameter sets with a positive equilibrium", n_eq)
    rep.true("Lem. 5.9", "reduced trace (5.13) holds on all samples", n_ok_a1 == n_eq, f"{n_ok_a1}/{n_eq}")
    rep.true("Thm. 5.10", "determinant formula (5.15) on all samples", n_ok_a3 == n_eq, f"{n_ok_a3}/{n_eq}")
    rep.true("Eq. (5.14)", "reduced = analytic = numerical Jacobian", n_ok_red == n_eq, f"{n_ok_red}/{n_eq}")
    rep.true("Thm. 5.10", "Routh-Hurwitz a1>0, a3>0, a1a2>a3 everywhere", n_ok_rh == n_eq, f"{n_ok_rh}/{n_eq}")
    rep.info("Thm. 5.10", "smallest relative RH margin (a1a2-a3)/(a1a2)", worst_rh)
    rep.true("Prop. 5.32", "order p_kap<p_E<p*<p_acc (sigma>amin+amax, L/v>amin)",
             n_ord_ok == n_ord, f"{n_ord_ok}/{n_ord}")
    rep.true("Lem. 5.25", "lambda strictly increasing, zero at f_crit", n_lam_ok == n_lam, f"{n_lam_ok}/{n_lam}")
    rep.true("App. D.6", "p*(m) strictly increasing (sigma>amin+amax)", n_mono_ok == n_mono, f"{n_mono_ok}/{n_mono}")
    # the hypothesis L/v > amin is needed: counterexample
    Qx = replace(P, L=500.0)   # L/v = 0.1 < amin
    mx = 2.0
    rep.true("Lem. 5.25 hyp.", "L/v < amin can give s* >= 1 (no f_crit)", s_star(mx, Qx) >= 1.0,
             s_star(mx, Qx), note="shows the hypothesis is not vacuous")


def sec_stability(rep: Report, P=PAR, full=False):
    rep.sec("Local stability, relaxation and the delay model (Sec. 5.2, App. B.2)")
    m = P.m0
    ps, pa = p_star(m, P), p_acc(m, P)
    table = [(0.4160, "8.05", "-0.000178", "5631", "9576"),
             (0.4300, "7.74", "-0.002826", "354", "569"),
             (0.4500, "7.33", "-0.005673", "176", "263"),
             (0.4800, "6.73", "-0.007947", "126", "169"),
             (0.5100, "6.18", "-0.004103", "244", "267"),
             (0.5170, "6.05", "-0.000206", "4846", "4863")]
    relax_rows = []
    for p, Vc, lc, rc, dc in table:
        x = equilibrium(p, m, P)
        Ja = jac_analytic(x, p, m, P)
        Jn = jac_numeric(lambda z: rhs_M(0, z, p, m, P), x)
        rep.true("Eq. (5.12)", f"analytic = numerical Jacobian, p={p}",
                 np.allclose(Ja, Jn, rtol=1e-5, atol=1e-8))
        lam, ev = slow_eig(p, m, P)
        rep.num("Tab. 5.1", f"V at p={p}", float(V_of_p(p)), Vc)
        rep.num("Tab. 5.1", f"Re lambda_slow, p={p}", lam.real, lc)
        rep.num("Tab. 5.1", f"relaxation [d], p={p}", 1 / abs(lam.real), rc)
        a1, a2, a3 = char_coeffs(Ja)
        rep.true("Thm. 5.10", f"RH at p={p}", a1 > 0 and a3 > 0 and a1 * a2 > a3)
        # delay model: rightmost root
        A0, A1, y = delay_linearisation(p, m, P)
        lD = rightmost_real_root(A0, A1, P.tau, lam.real)
        rep.num("Tab. 5.1", f"delay relaxation [d], p={p}", 1 / abs(lD), dc, tol=1.0)
        nrm = np.linalg.norm(A0, 2) + np.linalg.norm(A1, 2)
        rep.true("App. B.2", f"||A0||+||A1|| < 3 at p={p}", nrm < 3, nrm)
        c0 = count_roots(A0, A1, P.tau, 0.0)
        c1 = count_roots(A0, A1, P.tau, lD / 2)
        c2 = count_roots(A0, A1, P.tau, 1.5 * lD)
        rep.true("App. B.2", f"roots in Re>=0 / >=lD/2 / >=1.5lD: 0,0,1 (p={p})",
                 (c0, c1, c2) == (0, 0, 1), f"{c0},{c1},{c2}")
        relax_rows.append((p, 1 / abs(lam.real), 1 / abs(lD)))
    rep.num("Sec. 4.4", "delay relaxation up to x slower", max(r[2] / r[1] for r in relax_rows), "1.7")
    # dense scan: minimum relaxation
    pg = np.linspace(ps + 1e-4, pa - 1e-4, 4001)
    rt = np.array([1 / abs(slow_eig(p, m, P)[0].real) for p in pg])
    i = int(np.argmin(rt))
    rep.num("Rem. 5.11", "minimum relaxation on dense scan [d]", rt[i], "124", tol=1.0)
    rep.num("Rem. 5.11", "at p", pg[i], "0.49", tol=0.01)
    rep.num("Rem. 5.11", "at V", float(V_of_p(pg[i])), "6.6", tol=0.1)
    # near p*: two eigenvalues -> 0, third -> -0.587
    lam, ev = slow_eig(ps + 1e-7, m, P)
    evs = np.sort(ev.real)
    rep.true("Sec. 5.2.2", "two eigenvalues -> 0 as p -> p*", abs(evs[1]) < 1e-5 and abs(evs[2]) < 1e-5,
             f"{evs[1]:.2e}, {evs[2]:.2e}")
    rep.num("Sec. 5.2.2", "third eigenvalue near p*", evs[0], "-0.587")
    # nonlinear delay model decays at rate lambda_D (App. B.2 control)
    for p in (0.45, 0.48):
        A0, A1, y = delay_linearisation(p, m, P)
        lD = rightmost_real_root(A0, A1, P.tau, slow_eig(p, m, P)[0].real)
        y0 = y * np.array([1.0, 1.01, 0.99, 1.02])
        # measurement window: after the faster modes have decayed (exp(lD t) < 3e-3) and before
        # the perturbation reaches round-off (exp(lD t) > 1e-5); rate from successive differences,
        # which does not depend on the accuracy of the computed equilibrium
        t1 = round(min(math.log(3e-3) / lD, 2000.0), 1)
        t2 = round(min(math.log(1e-5) / lD, 3000.0), 1)
        res = dde_rk4(p, m, tuple(y0), 3001.0, dt=0.02, B_hist=y[0], P=P,
                      record=(t1, t1 + 1.0, t2, t2 + 1.0))
        dd = lambda t: np.linalg.norm(np.array(res["rec"][t + 1.0]) - np.array(res["rec"][t]))
        rate = math.log(dd(t2) / dd(t1)) / (t2 - t1)
        rep.num("App. B.2", f"nonlinear delay decay rate, p={p}", rate, f"{lD:.6g}", rel=5e-6,
                note="six significant figures")


def sec_global(rep: Report, P=PAR, full=False):
    rep.sec("Energy budget, Lyapunov bound and global collapse (Sec. 5.3)")
    m = P.m0
    rng = np.random.default_rng(7)
    # budget identity and the admissible range of the weight
    pts = np.column_stack([10 ** rng.uniform(-3, 5, 20000), 10 ** rng.uniform(-3, 5, 20000),
                           10 ** rng.uniform(-3, 4.5, 20000)])
    ok_id = ok_wa = ok_abs = True
    for x in pts[:5000]:
        p = rng.uniform(0, 1)
        r = rhs_M(0, x, p, m, P)
        H, F, f = x
        S = s_of(f) * H / (H + P.v)
        dWk = r[2] + P.kap * (r[0] + r[1])
        ok_id &= math.isclose(dWk, P.c * F * (p - p_kap(m, P)) - P.gA * H, rel_tol=1e-9, abs_tol=1e-9)
        a = rng.uniform(0, P.astar)
        dWa = r[2] + a * (r[0] + r[1])
        ok_wa &= math.isclose(dWa, P.L * S * (a - P.kap) + F * (P.c * p - P.gA - a * m) - P.gA * H,
                              rel_tol=1e-9, abs_tol=1e-9)
        ok_abs &= P.L * S * (a - P.kap) - P.gA * H <= -P.gA * H * H / (H + P.v) + 1e-9
    rep.true("Lem. 5.12", "budget identity (5.17) at 5000 random points", ok_id)
    rep.true("Lem. 5.14", "identity (5.18) for W_a", ok_wa)
    rep.true("Lem. 5.14", "absorption bound (5.20) for a <= a_star", ok_abs)
    a = P.astar * 1.001
    H, f = 1e-3, 1e8
    lhs = P.L * s_of(f) * H / (H + P.v) * (a - P.kap) - P.gA * H
    rep.true("Lem. 5.14", "bound fails for a > a_star (f large, H small)", lhs > 0, lhs)
    # other weights (Remark 5.16): numerical search over b != a
    p_other = search_other_weights(P, m)
    rep.num("Rem. 5.16", "best p with separate weights", p_other, "0.354", tol=0.0015)
    rep.num("Rem. 5.16", "corresponding V", float(V_of_p(p_other)), "9.5", tol=0.06)
    rep.num("Rem. 5.16", "beyond p_E by [%]", 100 * (p_other / p_E(m, P) - 1), "2", tol=0.5)
    # Theorem 5.17: collapse and monotone W for p < p_E, from A-D
    pe = p_E(m, P)
    ok_W = ok_col = True
    maxint = []
    for p in (0.30, 0.34):
        for key, x0 in STATES.items():
            sol = simulate_M(p, m, x0, 8000.0, rtol=1e-10, atol=1e-12, ration=True, stopN=1e-3)
            W = W_of(sol["x"], P.astar)
            ok_W &= bool(np.all(np.diff(W) <= 1e-6 * max(1, W[0])))
            ok_col &= sol["stopped"] is not None
            Fint = np.trapezoid(sol["x"][1], sol["t"]) if hasattr(np, "trapezoid") else np.trapz(sol["x"][1], sol["t"])
            bound = W[0] / (P.c * (pe - p))
            maxint.append(Fint / bound)
    rep.true("Thm. 5.17", "W nonincreasing along 8 runs with p < p_E", ok_W)
    rep.true("Thm. 5.17", "all 8 runs collapse (N < 1e-3)", ok_col)
    rep.true("Eq. (5.23)", "forager-days <= W(0)/(c(p_E - p))", max(maxint) <= 1.0, max(maxint))
    # residual food table (Tab. B.6) from the healthy state
    finf_claims = [(0.150, "104.5"), (0.200, "172.1"), (0.250, "216.0"), (0.300, "250.6"), (0.319, "262.2")]
    rows = []
    fc = f_crit(m, P)
    for p, cl in finf_claims:
        sol = simulate_M(p, m, HEALTHY, 6000.0, rtol=1e-10, atol=1e-12)
        fi = sol["x"][2, -1]
        rep.num("Tab. B.6", f"residual food at p={p}", fi, cl, tol=0.1)
        rep.true("Prop. 5.31", f"f_inf < f_crit at p={p}", fi < fc)
        rows.append((p, fi))
    sol = simulate_M(0.10, m, HEALTHY, 6000.0)
    rep.num("Fig. 5.1(b)", "p=0.10: stores run out after [d]", sol["exited"][0], "16", tol=0.5)
    rep.num("Fig. 5.1(b)", "bees alive at exit", sol["exited"][1], "5800", tol=100)
    solr = simulate_M(0.10, m, HEALTHY, 6000.0, ration=True, stopN=1.0)
    rep.num("Fig. 5.1(b)", "rationed: below one bee after [d]", solr["stopped"], "118", tol=0.5)
    # boundary of the two end states
    def exits(p):
        return simulate_M(p, m, HEALTHY, 6000.0, rtol=1e-10, atol=1e-12)["exited"] is not None
    pb = brentq(lambda p: 0.5 - float(exits(p)), 0.10, 0.15, xtol=1e-5)
    rep.num("Sec. 5.3.4", "end-state boundary p (exit below)", pb, "0.1157")
    rep.num("Sec. 5.3.4", "end-state boundary V", float(V_of_p(pb)), "19.8")
    rows.append((pb, 0.0))
    # residual food from A-D at p = 0.25
    res = []
    for key, x0 in STATES.items():
        sol = simulate_M(0.25, m, x0, 6000.0, rtol=1e-10, atol=1e-12)
        res.append(sol["x"][2, -1])
    rep.num("Sec. 5.3.4", "p=0.25, min residual over A-D [g]", min(res), "216.0", tol=0.1)
    rep.num("Sec. 5.3.4", "p=0.25, max residual over A-D [g]", max(res), "229.5", tol=0.1,
            note="(state D; the thesis figure 229.5 must come from other initial states)")
    # Remark 5.21: exit is a property of the state
    t_empties = []
    for V, cl_recover in [(7.3, True), (9.7, False), (11.0, False)]:
        mm = m + float(mu_of_V(V, P))
        s1 = simulate_M(float(p_of_V(V)), mm, (9000.0, 100.0, 5.0), 6000.0)
        rep.true("Rem. 5.21", f"(9000,100,5) at V={V}: stores reach 0 within ~0.1 d",
                 s1["exited"] is not None and s1["exited"][0] < 0.15,
                 None if s1["exited"] is None else round(s1["exited"][0], 4))
        s2 = simulate_M(float(p_of_V(V)), mm, (9000.0, 100.0, 5.0), 6000.0, ration=True, stopN=1.0)
        recovered = s2["stopped"] is None
        t_empties.append(s2["t_empty"])
        rep.true("Rem. 5.21", f"rationed, V={V}: {'recovers' if cl_recover else 'collapses'}",
                 recovered == cl_recover, f"t_empty={s2['t_empty']:.2f} d")
    s1 = simulate_M(1.0, m + float(mu_of_V(0.0)), (20000.0, 50.0, 1.0), 6000.0)
    rep.true("Rem. 5.21", "(20000,50,1) exits even at V=0", s1["exited"] is not None)
    s2 = simulate_M(1.0, m + float(mu_of_V(0.0)), (20000.0, 50.0, 1.0), 6000.0, ration=True, stopN=1.0)
    rep.true("Rem. 5.21", "rationed at V=0: recovers", s2["stopped"] is None, f"t_empty={s2['t_empty']:.2f} d")
    t_empties.append(s2["t_empty"])
    rep.num("Rem. 5.21", "shortest time on the empty comb [d]", min(t_empties), "0.14", tol=0.01)
    rep.num("Rem. 5.21", "longest time on the empty comb [d]", max(t_empties), "0.6", tol=0.05)
    # window table (Tab. B.7): time to H+F<1 in the window, m = m0 + mu(V)
    claims = {8.10: ("0.4136", ["16704", "16565", "16138", "14591"], "312.3"),
              8.50: ("0.3959", ["2207", "2103", "1902", "1580"], "303.8"),
              9.00: ("0.3749", ["1127", "1047", "924", "778"], "293.2"),
              9.50: ("0.3550", ["770", "704", "616", "536"], "282.6"),
              9.72: ("0.3466", ["678", "617", "539", "478"], "278.0")}
    wrows = []
    for V, (pc, tcl, fcl) in claims.items():
        p = float(p_of_V(V))
        mm = m + float(mu_of_V(V))
        rep.num("Tab. B.7", f"p at V={V}", p, pc)
        times, fres = [], []
        for (key, x0), cl in zip(STATES.items(), tcl):
            sol = simulate_M(p, mm, x0, 2e5, rtol=1e-10, atol=1e-12, stopN=1.0)
            times.append(sol["stopped"])
            fres.append(sol["x"][2, -1])
            rep.num("Tab. B.7", f"V={V}, state {key}: time to H+F<1 [d]",
                    sol["stopped"] if sol["stopped"] else np.nan, cl, rel=2e-3)
        rep.num("Tab. B.7", f"V={V}: residual food (state B) [g]", fres[1], fcl, tol=0.3)
        rep.info("Tab. B.7", f"V={V}: residual food spread over A-D [g]", max(fres) - min(fres))
        wrows.append((V, times, fres))


def search_other_weights(P, m):
    """
    Remark 5.16: W = f + a H + b F. Along any state, dW / N at the worst place
    (N -> 0 if a > kappa, N -> infinity otherwise) is
        g(u, s) = u (cp - gA - b m) - gA (1-u)
                  + (1-u) [ max(a-kappa, 0) (L/v) s + (b-a) (amin+amax - amax s - sigma u) ],
    linear in s = s(f) in [0,1]. W proves collapse at activity p if max g < 0.
    Returns the largest such p over (a, b).
    """
    ug = np.linspace(0.0, 1.0, 2001)

    def G(ab, p):
        a, b = ab
        best = -np.inf
        for s in (0.0, 1.0):
            g = (ug * (P.c * p - P.gA - b * m) - P.gA * (1 - ug)
                 + (1 - ug) * (max(a - P.kap, 0.0) * P.omega * s
                               + (b - a) * (P.asum - P.amax * s - P.sigma * ug)))
            best = max(best, float(np.max(g)))
        return best

    def feasible(p):
        starts = [(P.astar, P.astar), (P.astar, P.astar * 1.2), (P.astar * 1.1, P.astar * 1.3),
                  (P.astar * 0.9, P.astar * 1.4)]
        best = np.inf
        for s0 in starts:
            r = minimize(G, s0, args=(p,), method="Nelder-Mead",
                         options=dict(xatol=1e-9, fatol=1e-12, maxiter=4000))
            best = min(best, r.fun)
        return best < 0

    lo, hi = p_E(m, P) - 1e-6, p_star(m, P)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def sec_food_axis(rep: Report, P=PAR, full=False):
    rep.sec("Food axis, band lemma, transverse growth rate (Sec. 5.4)")
    m = P.m0
    um, up = band_limits(m, P)
    rep.num("Lem. 5.23", "u_-", um, "0.192")
    rep.num("Lem. 5.23", "u^+", up, "0.553")
    rep.num("Lem. 5.23", "H/F lower end 1/u^+ - 1", 1 / up - 1, "0.81")
    rep.num("Lem. 5.23", "H/F upper end 1/u_- - 1", 1 / um - 1, "4.2")
    rep.num("Cor. 5.24", "population bound L/(m u_-)", P.L / (m * um), "68000", tol=1000)
    # homogeneity formulas (Prop. 5.22) vs the rhs
    rng = np.random.default_rng(3)
    ok = True
    for _ in range(3000):
        H, F, f = 10 ** rng.uniform(-3, 5), 10 ** rng.uniform(-3, 5), 10 ** rng.uniform(-2, 4)
        p = rng.uniform(0, 1)
        r = rhs_M(0, (H, F, f), p, m, P)
        N = H + F
        u = F / N
        AN = P.L * s_of(f) / ((1 - u) * N + P.v)
        ok &= math.isclose((r[0] + r[1]) / N, AN * (1 - u) - m * u, rel_tol=1e-8, abs_tol=1e-10)
        du = r[1] / N - u * (r[0] + r[1]) / N
        ok &= math.isclose(du, (1 - u) * (alpha_of(f) - u * (P.sigma + AN + m)), rel_tol=1e-7, abs_tol=1e-10)
    rep.true("Prop. 5.22", "N- and u-equations (5.25)-(5.26) exact", ok)
    # u entering the band from outside
    sol = simulate_M(0.45, m, (20000.0, 20.0, 800.0), 200.0)
    u = sol["x"][1] / (sol["x"][0] + sol["x"][1])
    rep.true("Lem. 5.23", "u enters [u_-, u^+] from u=0.001", um - 1e-3 <= u[-1] <= up + 1e-3, u[-1])
    # lambda table
    fc = f_crit(m, P)
    rep.true("Lem. 5.25", "lambda(f_crit) = 0", abs(lam_of(fc, m, P)) < 1e-12)
    for f, cl in [(250, "-0.027012"), (313, "-0.000052"), (400, "0.038377"), (600, "0.115474"), (800, "0.168960")]:
        la = lam_of(float(f), m, P)
        rep.num("Tab. B.5", f"lambda analytic at f={f}", la, cl)
        us = u_star(float(f), m, P)
        N0 = 1e-4
        x0 = ((1 - us) * N0, us * N0, float(f))
        sol = solve_ivp(rhs_M, (0, 20), x0, method="LSODA", args=(1.0 if f > fc else 0.3, m, P),
                        rtol=1e-12, atol=1e-22)
        N = sol.y[0] + sol.y[1]
        meas = math.log(N[-1] / N[0]) / 20.0
        rep.num("Tab. B.5", f"lambda measured at f={f}", meas, cl)
        rep.true("Tab. B.5", f"|analytic - measured| < 5e-8 at f={f}", abs(meas - la) < 5e-8,
                 abs(meas - la))
    uc = u_star(fc, m, P)
    rate = (1 - uc) * (P.sigma + A_of(fc) + m)
    rep.num("Sec. 5.4.1", "shape relaxation rate at f_crit", rate, "0.59")
    rep.num("Rem. 5.28", "third eigenvalue -(1-u_c)(sigma+A+m0)", -rate, "-0.587")
    rep.num("App. D.3", "alpha sigma at most", P.asum * P.sigma, "0.375")
    rep.num("App. D.3", "(sigma+m0)^2 at least", (P.sigma + m) ** 2, "0.8172")
    # store drift G3 = c u_c (p - p*)
    ps = p_star(m, P)
    ok = True
    for p in np.linspace(0.2, 0.8, 13):
        G3 = P.c * p * uc - P.gA - P.kap * A_of(fc) * (1 - uc)
        ok &= abs(G3 - P.c * uc * (p - ps)) < 1e-13
    rep.true("Rem. 5.28", "store drift G3 = c u_c (p - p*) at the critical point", ok)
    rep.true("Rem. 5.28", "1/u_c = 1 + P_c", abs(1 / uc - 1 - P_c(m, P)) < 1e-12)
    # Theorem 5.26: attracting below f_crit, repelling above (numerical illustration)
    s_low = simulate_M(0.45, m, (0.6 * 1e-3, 0.4 * 1e-3, 200.0), 500.0)
    s_high = simulate_M(0.45, m, (0.6 * 1e-3, 0.4 * 1e-3, 450.0), 200.0)
    Nl = s_low["x"][0, -1] + s_low["x"][1, -1]
    Nh = s_high["x"][0, -1] + s_high["x"][1, -1]
    rep.true("Thm. 5.26", "small colony at f0=200 < f_crit decays", Nl < 1e-3, Nl)
    rep.true("Thm. 5.26", "small colony at f0=450 > f_crit grows", Nh > 1e-2, Nh)
    # Remark 5.30: separatrix numerics
    ok = True
    for p in (0.45, 0.48, 0.50):
        xe = equilibrium(p, m, P)
        a = simulate_M(p, m, (60.0, 25.0, 200.0), 4000.0, rtol=1e-9, atol=1e-9)
        b = simulate_M(p, m, (400.0, 150.0, 250.0), 4000.0, rtol=1e-9, atol=1e-9)
        c = simulate_M(p, m, (20000.0, 8000.0, 800.0), 4000.0, rtol=1e-9, atol=1e-9)
        dies = a["x"][0, -1] + a["x"][1, -1] < 1.0
        conv_b = np.linalg.norm(b["x"][:, -1] - xe) / np.linalg.norm(xe) < 1e-2
        conv_c = np.linalg.norm(c["x"][:, -1] - xe) / np.linalg.norm(xe) < 1e-2
        rep.true("Rem. 5.30", f"p={p}: (60,25,200) dies, others -> equilibrium", dies and conv_b and conv_c,
                 f"{dies},{conv_b},{conv_c}")
    # Proposition 5.33: W_c bound on {f <= f_crit}
    rng = np.random.default_rng(11)
    ac = a_c(m, P)
    ok = True
    for _ in range(5000):
        H, F = 10 ** rng.uniform(-3, 5, 2)
        f = rng.uniform(0, fc)
        p = rng.uniform(0, ps)
        r = rhs_M(0, (H, F, f), p, m, P)
        dW = r[2] + ac * (r[0] + r[1])
        ok &= dW <= -P.gA * H * H / (H + P.v) - P.c * (ps - p) * F + 1e-9
    rep.true("Prop. 5.33", "dW_c <= bound (5.32) on f <= f_crit (5000 points)", ok)
    # and a_c fails the global bound
    H, f = 1e-3, 1e8
    lhs = P.L * s_of(f) * H / (H + P.v) * (ac - P.kap) - P.gA * H
    rep.true("Prop. 5.33", "a_c fails the global bound of Lemma 5.14", lhs > 0)


def sec_channels(rep: Report, P=PAR, full=False):
    rep.sec("The two channels (Ch. 6, App. A.2)")
    m0 = P.m0
    mb = mu_bound(P)
    # trajectory
    rep.num("Sec. 6.1", "m at V=0", m0 + float(mu_of_V(0.0)), "0.154014")
    rep.num("Sec. 6.1", "m at V=16.6", m0 + mb, "0.154362")
    rep.num("Sec. 6.1", "m at V=20", m0 + float(mu_of_V(20.0)), "0.154341")
    rep.num("Sec. 6.1", "p at V=20", float(p_of_V(20.0)), "0.113")
    rep.num("Sec. 6.1", "excursion of m on [0,20]", mb - float(mu_of_V(0.0)), "3.5e-4")
    # Tab. A.3
    for V, pc, hfc, npc, muc, rc in [(6.00, "0.520", "8.43e-5", "1.56", "1.32e-4", "0.085"),
                                     (8.07, "0.415", "1.60e-4", "1.24", "1.99e-4", "0.129"),
                                     (12.60, "0.253", "4.28e-4", "0.76", "3.25e-4", "0.211"),
                                     (16.60, "0.164", "7.37e-4", "0.49", "3.62e-4", "0.235"),
                                     (20.00, "0.113", "1.01e-3", "0.34", "3.41e-4", "0.221")]:
        p = float(p_of_V(V))
        rep.num("Tab. A.3", f"p({V})", p, pc)
        rep.num("Tab. A.3", f"HF at V={V}", float(HF_of_p(p)), hfc)
        rep.num("Tab. A.3", f"nu p at V={V}", P.nu * p, npc)
        rep.num("Tab. A.3", f"mu at V={V}", float(mu_of_p(p)), muc)
        rep.num("Tab. A.3", f"mu/m0 [%] at V={V}", 100 * float(mu_of_p(p)) / m0, rc)
    # Remark 4.1 / Q1: thresholds with mu carried
    mfun = lambda p: m0 + float(mu_of_p(p, P))
    names = {"acc": p_acc, "star": p_star, "E": p_E}
    shift_p, shift_V = [], []
    for k, T in names.items():
        p0 = T(m0, P)
        p1 = threshold_with_m_of_p(k, mfun, P)
        shift_p.append(p1 - p0)
        shift_V.append(float(V_of_p(p0) - V_of_p(p1)))
    rep.true("Rem. 4.1", "mu raises each threshold by at most 0.0005 in p",
             max(shift_p) <= 0.0005 and min(shift_p) > 0, f"{max(shift_p):.5f}")
    rep.num("Rem. 4.1", "largest shift in hornets", max(shift_V), "0.01", tol=0.006)
    ps_mu = threshold_with_m_of_p("star", mfun, P)
    rep.num("Sec. 6.2 Q1", "p* with mu carried", ps_mu, "0.4155")
    rep.num("Sec. 6.2 Q1", "shift of V* [hornets]", float(V_of_p(p_star(m0)) - V_of_p(ps_mu)), "0.01", tol=0.005)
    sh = []
    for k in (3.0, 12.0):
        pk = threshold_with_m_of_p("star", lambda p: m0 + float(mu_of_p(p, P, k)), P)
        sh.append(float(V_of_p(p_star(m0)) - V_of_p(pk)))
    rep.num("Sec. 6.2 Q1", "HF x3: shift of V* [hornets]", sh[0], "0.03", tol=0.006)
    rep.num("Sec. 6.2 Q1", "HF x12: shift of V* [hornets]", sh[1], "0.10", tol=0.006)
    # Q2: capture rate that moves an edge by one hornet
    Vs = float(V_of_p(p_star(m0)))
    Vt = Vs - 1
    pt = float(p_of_V(Vt))
    mt = brentq(lambda mm: p_star(mm, P) - pt, m0, 1.0)
    dm = mt - m0
    rep.num("Sec. 6.2 Q2", "V* - 1", Vt, "7.07")
    rep.num("Sec. 6.2 Q2", "required Delta m", dm, "0.025")
    rep.num("Sec. 6.2 Q2", "trips per day nu p", P.nu * pt, "1.39")
    cap = dm / (P.nu * pt)
    rep.num("Sec. 6.2 Q2", "required capture rate [%]", 100 * cap, "1.8")
    rep.num("Sec. 6.2 Q2", "one flight in", 1 / cap, "56", tol=1.0)
    rep.num("Sec. 6.2 Q2", "times highest fitted rate h0", cap / P.h0, "9", tol=0.5)
    Fst = equilibrium(pt, m0, P)[1]
    rep.num("Sec. 6.2 Q2", "foragers at resting state (V*-1)", Fst, "6900", tol=100)
    rep.num("Sec. 6.2 Q2", "bees lost per day", dm * Fst, "170", tol=6)
    for T, lab, cl in [(p_acc, "V_acc", "1.75"), (p_E, "V_E", "1.9")]:
        V0 = float(V_of_p(T(m0, P)))
        ptt = float(p_of_V(V0 - 1))
        mtt = brentq(lambda mm: T(mm, P) - ptt, m0, 2.0)
        rep.num("Sec. 6.2 Q2", f"capture rate for {lab} [%]", 100 * (mtt - m0) / (P.nu * ptt), cl)
    # Q3
    ms = m_star(P)
    rep.num("Sec. 6.2 Q3", "capture rate at full traffic [%]", 100 * (ms - m0) / P.nu, "10.3")
    rep.num("Eq. (6.4)", "(m* - m0) / max mu", (ms - m0) / mb, "851", tol=1.0)
    rep.num("Sec. 6.2 Q3", "nu h0 [1/d]", P.nu * P.h0, "0.006")
    rep.num("Sec. 6.2 Q3", "(m* - m0)/(nu h0)", (ms - m0) / (P.nu * P.h0), "51", tol=0.5)
    rep.num("Sec. 6.2 Q3", "zeta e", P.zeta * math.e, "17", tol=0.5)
    rep.true("Sec. 6.2 Q3", "851 = 51 x 17 exactly", abs((ms - m0) / mb - (ms - m0) / (P.nu * P.h0) * P.zeta * math.e) < 1e-9)
    rep.num("Sec. 6.2 Q3", "(0.401 - m0)/max mu", (m_acc_boundary(P) - m0) / mb, "680", tol=5)
    # video captures (App. A.2)
    cpd = 126 / 10
    rep.num("App. A.2", "captures per day", cpd, "12.6")
    rep.num("App. A.2", "foragers needed at the bound", cpd / mb, "35000", tol=500)
    mu_lo, mu_hi = cpd / 11000, cpd / 3000
    rep.num("App. A.2", "mu_obs at 11000 foragers", mu_lo, "1.1e-3")
    rep.num("App. A.2", "mu_obs at 3000 foragers", mu_hi, "4.2e-3")
    rep.num("App. A.2", "mu_obs / m0 low [%]", 100 * mu_lo / m0, "0.7")
    rep.num("App. A.2", "mu_obs / m0 high [%]", 100 * mu_hi / m0, "2.7")
    rep.num("App. A.2", "times the bound, low", mu_lo / mb, "3", tol=0.5)
    rep.num("App. A.2", "times the bound, high", mu_hi / mb, "12", tol=0.5)
    rep.num("App. A.2", "shortfall factor, high mu_obs", (ms - m0) / mu_hi, "73", tol=1)
    rep.num("App. A.2", "shortfall factor, low mu_obs", (ms - m0) / mu_lo, "270", tol=2)
    hf_mean = 126 / 299039
    rep.num("App. A.2", "mean homing failure", hf_mean, "4.2e-4")
    p_at = brentq(lambda p: float(HF_of_p(p)) - hf_mean, 0, 1)
    rep.num("App. A.2", "activity where HF reaches it", p_at, "0.26")
    rep.num("App. A.2", "trips per day there", P.nu * p_at, "0.77", tol=0.01)
    rep.num("App. A.2", "implied foragers", 30000 / (P.nu * p_at), "39000", tol=500)
    rep.num("App. A.2", "598078 / 2", 598078 / 2, "299039")
    # BEEHAVE comparison
    rep.num("Sec. 7.2", "seconds of flight for 1e-5/s to equal max mu", mb / 1e-5, "36", tol=0.5)


def sec_sensitivity(rep: Report, P=PAR, full=False):
    rep.sec("Sensitivity: beta, forage quality, theta, m0, (p,m) plane (Sec. 6.3, App. A-B)")
    m0 = P.m0
    th = [p_acc(m0), p_star(m0), p_E(m0)]
    tab = {0.087: ("7.56", "10.10", "12.18"), 0.107: ("6.15", "8.22", "9.91"), 0.109: ("6.04", "8.07", "9.73"),
           0.120: ("5.48", "7.33", "8.83"), 0.131: ("5.02", "6.71", "8.09")}
    for bta, cls in tab.items():
        for nm, pv, cl in zip(("V_acc", "V*", "V_E"), th, cls):
            rep.num("Tab. 6.1", f"{nm} at beta={bta}", float(V_of_p(pv, beta=bta)), cl)
    rep.num("Tab. 6.1", "beta -20 %", 0.1098 * 0.8, "0.087", tol=0.001)
    rep.num("Tab. 6.1", "beta +20 %", 0.1098 * 1.2, "0.131", tol=0.001)
    # forage quality: all thresholds in p scale as 1/c
    Pc = replace(P, c=0.12)
    rep.true("Rem. 6.1", "thresholds in p scale exactly as 1/c",
             all(abs(T(m0, Pc) * Pc.c - T(m0, P) * P.c) < 1e-14 for T in (p_acc, p_star, p_E, p_kap)))
    rep.num("Rem. 6.1", "20 % richer site buys [hornets]", math.log(1.2) / P.beta, "1.7")
    rep.num("Rem. 6.1", "twice as rich buys [hornets]", math.log(2.0) / P.beta, "6.4")
    # theta bracket
    tclaims = {0.0: ("0.5181", "6.03", "0.4155", "8.06", "0.3469", "9.71"),
               0.25: ("0.4785", "6.76", "0.3685", "9.16", "0.2985", "11.09"),
               0.5: ("0.4316", "7.71", "0.3128", "10.66", "0.2423", "13.01"),
               0.75: ("0.3753", "8.99", "0.2452", "12.90", "0.1763", "15.92"),
               1.0: ("0.3059", "10.87", "0.1590", "16.87", "0.0975", "21.35")}
    trows = []
    for theta, cl in tclaims.items():
        mf = lambda p, th_=theta: m0 * ((1 - th_) + th_ * p) + float(mu_of_p(p, P))
        vals = []
        for i, k in enumerate(("acc", "star", "E")):
            pv = threshold_with_m_of_p(k, mf, P)
            rep.num("Tab. 6.2", f"p_{k} at theta={theta}", pv, cl[2 * i])
            rep.num("Tab. 6.2", f"V_{k} at theta={theta}", float(V_of_p(pv)), cl[2 * i + 1])
            vals.append(pv)
        trows.append((theta, vals))
    # uniqueness claim: slope of each edge along m(p) below 0.4 in p (slope of T(m(p)) < 1)
    ok = True
    for theta in np.linspace(0, 1, 11):
        mf = lambda p: m0 * ((1 - theta) + theta * p) + float(mu_of_p(p, P))
        pg = np.linspace(0.02, 1, 500)
        for T in (p_acc, p_star, p_E):
            vals = np.array([T(mf(p)) for p in pg])
            ok &= np.max(np.abs(np.diff(vals) / np.diff(pg))) < 0.4
    rep.true("Sec. 6.3", "each edge grows along m(p) with slope < 0.4 (unique fixed point)", ok)
    # tilde closed forms at theta = 1 without HF
    pE_t = P.gA / (P.c - P.astar * m0)
    rep.num("Tab. 6.2 cap.", "tilde p_E", pE_t, "0.0967")
    rep.num("Tab. 6.2 cap.", "tilde V_E", float(V_of_p(pE_t)), "21.4")
    k = P.gA * m0 / (P.c - P.kap * m0)
    Qt = max(np.roots([P.amin - k, P.amin - P.sigma - 2 * k, -k]).real)
    pacc_t = P.gA * (1 + Qt) / (P.c - P.kap * m0)
    rep.num("Tab. 6.2 cap.", "tilde p_acc (closed form)", pacc_t, "0.3051")
    rep.num("Tab. 6.2 cap.", "tilde V_acc", float(V_of_p(pacc_t)), "10.9")
    pacc_fp = threshold_with_m_of_p("acc", lambda p: m0 * p, P)
    rep.true("Tab. 6.2 cap.", "closed form = fixed point without HF", abs(pacc_fp - pacc_t) < 1e-10)
    diffs = []
    for i, T in enumerate(("acc", "star", "E")):
        a = threshold_with_m_of_p(T, lambda p: m0 * p + float(mu_of_p(p, P)), P)
        b = threshold_with_m_of_p(T, lambda p: m0 * p, P)
        diffs.append(abs(float(V_of_p(a) - V_of_p(b))))
    rep.true("Tab. 6.2 cap.", "dropping HF at theta=1 moves each result < 0.1 hornets", max(diffs) < 0.1, max(diffs))
    th95 = brentq(lambda t: float(V_of_p(threshold_with_m_of_p(
        "E", lambda p: m0 * ((1 - t) + t * p) + float(mu_of_p(p, P)), P))) - 20.0, 0.5, 1.0)
    rep.num("Sec. 6.3", "theta at which V_E = 20", th95, "0.95", tol=0.01)
    # m0 range table (Tab. A.2)
    for fs, mm, cl in [(8.8, 0.114, ("7.56", "9.96", "11.86", "281.6", "961")),
                       (7.5, 0.133, ("6.80", "9.01", "10.78", "297.4", "909")),
                       (6.7, 0.149, ("6.21", "8.28", "9.97", "309.5", "865")),
                       (6.5, 0.154, ("6.04", "8.07", "9.73", "313.1", "851"))]:
        Pm = replace(P, m0=mm)
        vals = [float(V_of_p(p_acc(mm, Pm))), float(V_of_p(p_star(mm, Pm))), float(V_of_p(p_E(mm, Pm))),
                f_crit(mm, Pm), (m_star(Pm) - mm) / mu_bound(Pm)]
        for nm, v_, c_ in zip(("V_acc", "V*", "V_E", "f_crit", "ratio"), vals, cl):
            rep.num("Tab. A.2", f"{nm}, flightspan {fs} d", v_, c_, tol=1.0 if nm == "ratio" else None)
        rep.num("Tab. A.2", f"1/{fs} rounded", 1 / fs, str(mm), tol=0.0006)
    # plane table (Tab. B.4)
    for mm, cl in [(0.154, ("0.4151", "0.5178", "0.1027", "0.3464")), (0.2, ("0.5036", "0.6083", "0.1047", "0.4290")),
                   (0.25, ("0.5990", "0.7062", "0.1072", "0.5187")), (0.3, ("0.6939", "0.8037", "0.1098", "0.6085")),
                   (0.35, ("0.7884", "0.9010", "0.1126", "0.6982")), (0.401, ("0.8846", "1.0000", "0.1154", "0.7898")),
                   (0.4624, ("1.0000", "1.1188", "0.1188", "0.9000"))]:
        vals = [p_star(mm), p_acc(mm), p_acc(mm) - p_star(mm), p_E(mm)]
        for nm, v_, c_ in zip(("p*", "p_acc", "width", "p_E"), vals, cl):
            rep.num("Tab. B.4", f"{nm} at m={mm}", v_, c_, tol=1.5e-4 if mm in (0.401, 0.4624) else None)


def sec_season(rep: Report, P=PAR, full=False):
    rep.sec("Season store-balance load (Sec. 6.4, Tab. 6.3)")
    m0 = P.m0
    claims = {(70, "M"): ("5.53", "5.67", "6.01", "5.92"), (70, "D"): ("5.50", "5.59", "5.80", "6.14"),
              (150, "M"): ("5.83", "6.23", "6.78", "6.79"), (150, "D"): ("5.80", "6.14", "6.54", "6.49")}

    def dstore_M(V, x0, T):
        p = float(p_of_V(V))
        mm = m0 + float(mu_of_V(V))
        sol = solve_ivp(rhs_M, (0, T), x0, method="LSODA", args=(p, mm, P), rtol=1e-9, atol=1e-9)
        return sol.y[2, -1] - x0[2]

    def dstore_D(V, x0, T):
        p = float(p_of_V(V))
        mm = m0 + float(mu_of_V(V))
        H0, F0, f0 = x0
        B0 = P.L * s_of(f0) * H0 / (H0 + P.v) / P.phi
        r = dde_rk4(p, mm, (B0, H0, F0, f0), T, dt=0.02, B_hist=B0, P=P)
        return r["final"][3] - f0

    out = {}
    allv = []
    for (T, mod), cls in claims.items():
        fun = dstore_M if mod == "M" else dstore_D
        for (key, x0), cl in zip(STATES.items(), cls):
            grid = np.array([fun(V, x0, T) for V in range(21)])
            if grid is not None:
                nsc = int(np.sum(np.sign(grid[:-1]) != np.sign(grid[1:])))
                rep.true("Tab. 6.3", f"one sign change on V=0..20 ({mod}, T={T}, {key})", nsc == 1, nsc)
                k = int(np.argmax(np.sign(grid[:-1]) != np.sign(grid[1:])))
                lo, hi = k, k + 1
            else:
                lo, hi = 4.0, 8.0
            Vb = brentq(lambda V: fun(V, x0, T), lo, hi, xtol=1e-4)
            out[(T, mod, key)] = Vb
            allv.append(Vb)
            rep.num("Tab. 6.3", f"store-balance load, {'(M)' if mod == 'M' else 'delay'}, T={T}, {key}",
                    Vb, cl, tol=0.02)
    rep.true("Sec. 6.4", "all store-balance loads within [5.5, 6.8]",
             min(allv) >= 5.495 and max(allv) <= 6.805, f"{min(allv):.2f}-{max(allv):.2f}")
    rep.true("Sec. 6.4", "V_acc = 6.04 inside that range", min(allv) <= float(V_of_p(p_acc(m0))) <= max(allv))
    dmax = max(abs(out[(T, 'M', k)] - out[(T, 'D', k)]) for T in (70, 150) for k in STATES)
    rep.true("Sec. 6.4", "delay moves the load by at most 0.3 hornets", dmax <= 0.3 + 1e-3, dmax)
    rep.true("Sec. 6.4", "within a hornet of V_acc", max(abs(v - float(V_of_p(p_acc(m0)))) for v in allv) < 1.0)


def sec_delay(rep: Report, P=PAR, full=False):
    rep.sec("Delay model: collapse, W_D, finite-horizon thresholds (Thm. 5.20, App. B.2, D.7)")
    m0 = P.m0
    # Theorem 5.20 / App. D.7: W_D never increases at V = 11, collapse within 4000 d
    V = 11.0
    p = float(p_of_V(V))
    mm = m0 + float(mu_of_V(V))
    t0 = time.time()
    r = dde_rk4(p, mm, (0.0, 16000.0, 8000.0, 800.0), 4000.0, dt=0.02, B_hist=0.0, P=P, track_W=True)
    B, H, F, f = r["final"]
    rep.true("App. D.7", "delay model collapses within 4000 d at V=11", H + F < 1.0, H + F)
    rep.true("App. D.7", "W_D never increases along the computed solution", r["dWmax"] <= 1e-9, r["dWmax"])
    rep.true("Thm. 5.20", "f stays >= 0 along that solution", f >= 0, f)
    rep.info("App. D.7", "runtime [s]", time.time() - t0)
    # pupae bookkeeping identity (Remark D.1) at random states: d/dt[f + kap(H+F+Pi)] with Pi-dot
    rng = np.random.default_rng(5)
    ok = True
    for _ in range(2000):
        Bv, Hv, Fv, fv, Bd = 10 ** rng.uniform(0, 5, 5)
        pv = rng.uniform(0, 1)
        s = s_of(fv)
        rec = Hv * (P.amin + P.amax * (1 - s) - P.sigma * Fv / (Hv + Fv))
        dH, dF = P.phi * Bd - rec, rec - m0 * Fv
        df = P.c * pv * Fv - P.gA * (Hv + Fv) - P.gB * Bv
        dPi = P.phi * Bv - P.phi * Bd
        lhs = df + P.kap * (dH + dF + dPi)
        ok &= math.isclose(lhs, P.c * Fv * (pv - p_kap(m0)) - P.gA * Hv, rel_tol=1e-9, abs_tol=1e-8)
    rep.true("Rem. D.1", "budget identity in the delay model with pupae", ok)
    if not full:
        rep.info("Tab. B.3", "finite-horizon thresholds skipped", "run without --quick")
        return
    # Table B.3: finite-horizon survival thresholds
    claims = {"delay": ("0.3879", "0.4031", "0.4104"), "tau0": ("0.3981", "0.4084", "0.4129"),
              "M3": ("0.3983", "0.4086", "0.4130")}
    res = {}
    for model in ("delay", "tau0", "M3"):
        t0 = time.time()
        th = survival_thresholds(model, P)
        res[model] = th
        for T, cl in zip((1000, 2000, 4000), claims[model]):
            rep.num("Tab. B.3", f"threshold {model}, T={T}", th[T], cl)
        rep.info("Tab. B.3", f"runtime {model} [s]", time.time() - t0)
    for T in (1000, 2000, 4000):
        d = 100 * (res["M3"][T] / res["delay"][T] - 1)
        rep.info("Tab. B.3", f"deviation of (M) vs delay at T={T} [%]", d)
    th0 = survival_thresholds("M3f0", P)
    sh = max(100 * abs(th0[T] / res["M3"][T] - 1) for T in (1000, 2000, 4000))
    rep.num("App. B.2", "(M) with f0 = 0 instead of 800: largest shift [%]", sh, "0.09",
            note="(at T=1000; 0.02 % at 2000, 0.005 % at 4000)")
    rep.true("Sec. 4.4", "reduction within 0.6 % of delay model at 4000 d",
             abs(res["M3"][4000] / res["delay"][4000] - 1) <= 0.0065)
    rep.true("App. B.2", "tau=0 ODE within 0.02 percentage points of (M)",
             all(abs(res["tau0"][T] - res["M3"][T]) / res["delay"][T] * 100 <= 0.1 for T in (1000, 2000, 4000)))
    rep.true("App. B.2", "thresholds rise with horizon towards p*",
             all(res[k][1000] < res[k][2000] < res[k][4000] < p_star(m0) for k in res))
    # step-size check (Tab. B.3 caption): the delay threshold at T = 2000 is 0.4031 at dt = 0.04, 0.02, 0.01
    for dt in (0.04, 0.01):
        th = survival_thresholds("delay", P, horizons=(2000,), dt=dt)[2000]
        rep.num("Tab. B.3", f"delay threshold T=2000 at dt={dt}", th, "0.4031")


def rk4_M_scalar(p, m, x0, T, dt=0.02, P=PAR):
    """Fixed-step RK4 for (M) on plain floats (fast in pure Python)."""
    L, amin, amax, sig, v, b2, c, gA, kL = P.L, P.amin, P.amax, P.sigma, P.v, P.b * P.b, P.c, P.gA, P.kap * P.L
    cp = c * p

    def r(H, F, f):
        f2 = f * f
        s = f2 / (f2 + b2)
        N = H + F
        S = s * H / (H + v)
        rec = H * (amin + amax * (1.0 - s) - sig * F / (N + 1e-300))
        return L * S - rec, rec - m * F, cp * F - gA * N - kL * S

    H, F, f = x0
    h2 = 0.5 * dt
    w = dt / 6.0
    for _ in range(int(round(T / dt))):
        a = r(H, F, f)
        b = r(H + h2 * a[0], F + h2 * a[1], f + h2 * a[2])
        c_ = r(H + h2 * b[0], F + h2 * b[1], f + h2 * b[2])
        d = r(H + dt * c_[0], F + dt * c_[1], f + dt * c_[2])
        H += w * (a[0] + 2 * b[0] + 2 * c_[0] + d[0])
        F += w * (a[1] + 2 * b[1] + 2 * c_[1] + d[1])
        f += w * (a[2] + 2 * b[2] + 2 * c_[2] + d[2])
    return H, F, f


def survival_thresholds(model, P=PAR, lo=0.30, hi=0.70, nbis=20, horizons=(1000, 2000, 4000), dt=0.02):
    """
    Finite-horizon survival threshold (App. B.2): fixed-step RK4 from the
    published initial state (B,H,F,f) = (0,16000,8000,0) at m = m0 (the
    three-dimensional model starts with f0 = 800); a colony persists if
    H + F > 200 at T. 20 bisection steps on p in [0.30, 0.70].
    """
    m0 = P.m0

    def persists(p, T):
        if model in ("M3", "M3f0"):
            f0 = 800.0 if model == "M3" else 0.0
            H, F, f = rk4_M_scalar(p, m0, (16000.0, 8000.0, f0), T, dt=dt, P=P)
            return H + F > 200
        tau = P.tau if model == "delay" else 0.0
        B, H, F, f = dde_rk4(p, m0, (0.0, 16000.0, 8000.0, 0.0), T, dt=dt, tau=tau, B_hist=0.0, P=P)["final"]
        return H + F > 200

    out = {}
    for T in horizons:
        a, b = lo, hi
        for _ in range(nbis):
            mid = 0.5 * (a + b)
            if persists(mid, float(T)):
                b = mid
            else:
                a = mid
        out[T] = 0.5 * (a + b)
    return out


# =============================================================================
# 9. FIGURES
# =============================================================================

TEXTWIDTH = 6.3   # inches, 16 cm


def _mpl():
    import matplotlib
    if "ipykernel" not in sys.modules:      # keep the inline backend inside Jupyter
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "lines.linewidth": 1.3,
    })
    return plt


COL = dict(H="#1f5f8b", F="#d1495b", f="#e0a100", N="#2e2e2e", band="#cfe3f2", window="#bbbbbb",
           collapse="#f4c7c3", acc="#d9f0d3", edge="#1f5f8b", bound="#b2182b", traj="#e66101")


def _top_V_axis(ax, P=PAR, ticks=(0, 2, 4, 6, 8, 10, 12, 15, 20)):
    sec = ax.secondary_xaxis("top", functions=(lambda p: -np.log(np.clip(p, 1e-9, None)) / P.beta,
                                               lambda V: np.exp(-P.beta * V)))
    sec.set_xlabel("hornet load $V$")
    return sec


def _shade_regions(ax, P=PAR, ymax=None):
    m = P.m0
    pe, ps, pa = p_E(m), p_star(m), p_acc(m)
    ax.axvspan(0, pe, color=COL["collapse"], lw=0, zorder=0)
    ax.axvspan(pe, ps, facecolor="none", edgecolor=COL["window"], hatch="////", lw=0, zorder=0)
    ax.axvspan(pa, 1, color=COL["acc"], lw=0, zorder=0)


def fig_band(outdir, P=PAR):
    """Fig. 5.2: branch of equilibria across the existence band."""
    plt = _mpl()
    m = P.m0
    ps, pa, pe = p_star(m), p_acc(m), p_E(m)
    pg = np.linspace(ps + 1e-6, pa - 1e-6, 2000)
    X = np.array([equilibrium(p, m) for p in pg])
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    for ax in axs:
        _shade_regions(ax, P)
        ax.set_xlim(0.30, 0.56)
        ax.set_xlabel("foraging activity $p$")
        _top_V_axis(ax)
    ax = axs[0]
    ax.plot(pg, X[:, 0] + X[:, 1], color=COL["N"])
    ax.plot([0.30, 0.56], [0, 0], color="k", lw=3, solid_capstyle="butt")
    ax.set_ylabel("adults $H^*+F^*$")
    ax.set_ylim(-1500, 45000)
    ax.text(0.425, 2500, "food axis $H=F=0$", fontsize=7)
    ax.text(0.02, 0.93, "(a)", transform=ax.transAxes, fontsize=9)
    ax = axs[1]
    ax.semilogy(pg, X[:, 2], color=COL["f"])
    ax.axhline(f_crit(m), ls="--", color="k", lw=0.8)
    ax.text(0.305, f_crit(m) * 1.15, r"$f_{\rm crit}$", fontsize=8)
    ax.set_ylim(100, 2e4)
    ax.set_ylabel("stores $f^*$ [g]")
    ax.text(0.02, 0.93, "(b)", transform=ax.transAxes, fontsize=9)
    for ax in axs:
        for pv, lab in [(pe, "$p_E$"), (ps, "$p^*$"), (pa, r"$p_{\rm acc}$")]:
            ax.axvline(pv, color="k", lw=0.5, ls=":")
    fig.tight_layout()
    return _save(fig, outdir, "band")


def fig_pm(outdir, P=PAR):
    """Fig. 6.1: the (p, m) plane and the hornet trajectory."""
    plt = _mpl()
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    mg = np.linspace(0.02, 0.62, 600)
    ps = np.array([p_star(m) for m in mg])
    pa = np.array([p_acc(m) for m in mg])
    pe = np.array([p_E(m) for m in mg])
    ax.fill_betweenx(mg, 0, pe, color=COL["collapse"], lw=0)
    ax.fill_betweenx(mg, pe, ps, facecolor="none", edgecolor=COL["window"], hatch="////", lw=0)
    ax.fill_betweenx(mg, ps, pa, color=COL["band"], lw=0)
    ax.fill_betweenx(mg, pa, 1.0, color=COL["acc"], lw=0)
    ax.plot(ps, mg, color=COL["edge"], lw=1)
    ax.plot(pa, mg, color=COL["edge"], lw=1, ls="--")
    ax.plot(pe, mg, color=COL["bound"], lw=1, ls="-.")
    Vg = np.linspace(0, 20, 400)
    ax.plot(p_of_V(Vg), P.m0 + mu_of_V(Vg), color=COL["traj"], lw=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 0.6)
    ax.set_xlabel("foraging activity $p$")
    ax.set_ylabel("forager mortality $m$ [d$^{-1}$]")
    ax.text(0.06, 0.52, "collapse\nproved", fontsize=7)
    ax.text(0.83, 0.1, "stores\ngrow", fontsize=7)
    ax.plot(1, m_star(), "ko", ms=3, clip_on=False)
    ax.text(0.9, m_star() - 0.05, "$m^*$", fontsize=8)
    ax.text(0.47, 0.3, "band", fontsize=7, rotation=38)
    ax.text(0.30, 0.36, "window", fontsize=7, rotation=38)
    ax.set_title("(a)", loc="left", fontsize=9)
    ax = axs[1]
    Vg = np.linspace(0, 20, 800)
    ax.plot(p_of_V(Vg), 1e4 * mu_of_V(Vg), color=COL["traj"], lw=1.6)
    for V, lab, off in [(float(V_of_p(p_acc(P.m0))), r"$V_{\rm acc}$", (4, 3)), (float(V_of_p(p_star(P.m0))), "$V^*$", (4, 3)),
                        (float(V_of_p(p_E(P.m0))), "$V_E$", (4, 3)), (16.6, "$V=16.6$", (5, 2)), (20.0, "$V=20$", (-4, -11))]:
        ax.plot(p_of_V(V), 1e4 * mu_of_V(V), "o", color="k", ms=3)
        ax.annotate(lab, (p_of_V(V), 1e4 * mu_of_V(V)), textcoords="offset points", xytext=off, fontsize=7)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("foraging activity $p$")
    ax.set_ylabel(r"$m-m_0$ [$10^{-4}$ d$^{-1}$]")
    ax.set_title("(b)  trajectory, $m$-axis stretched", loc="left", fontsize=9)
    fig.tight_layout()
    return _save(fig, outdir, "pm")


def fig_endstates(outdir, P=PAR):
    """Fig. 5.1: the two end states from the healthy state."""
    plt = _mpl()
    m = P.m0
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    sol = simulate_M(0.25, m, HEALTHY, 250.0)
    ax = axs[0]
    ax.plot(sol["t"], sol["x"][0] + sol["x"][1], color=COL["N"], label="adults $H+F$")
    ax.set_ylabel("adults")
    ax2 = ax.twinx()
    ax2.plot(sol["t"], sol["x"][2], color=COL["f"], label="stores $f$")
    ax2.axhline(f_crit(m), color=COL["f"], ls=":", lw=0.9)
    ax2.text(170, f_crit(m) + 25, r"$f_{\rm crit}$", fontsize=7, color=COL["f"])
    ax2.set_ylim(0, 1400)
    ax.set_ylim(0, None)
    ax2.set_ylabel("stores [g]", color=COL["f"])
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("time [d]")
    ax.set_title(f"(a)  $p=0.25$, $V={float(V_of_p(0.25)):.1f}$", loc="left", fontsize=9)
    s1 = simulate_M(0.10, m, HEALTHY, 200.0)
    s2 = simulate_M(0.10, m, HEALTHY, 200.0, ration=True, stopN=1.0)
    ax = axs[1]
    ax.plot(s2["t"], s2["x"][0] + s2["x"][1], color=COL["N"], ls="--", label="with rationing")
    ax.plot(s1["t"], s1["x"][0] + s1["x"][1], color=COL["N"], label="model (M)")
    te, Ne = s1["exited"]
    ax.plot(te, Ne, "o", color=COL["bound"], ms=4)
    ax.annotate("stores empty", (te, Ne), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(s1["t"], s1["x"][2], color=COL["f"])
    ax2.set_ylabel("stores [g]", color=COL["f"])
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("time [d]")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"(b)  $p=0.10$, $V={float(V_of_p(0.10)):.1f}$", loc="left", fontsize=9)
    fig.tight_layout()
    return _save(fig, outdir, "endstates")


def fig_calibration(outdir, P=PAR):
    """Calibration chain (Section 4.3): data -> fitted function -> model term. Not printed in the thesis."""
    plt = _mpl()
    fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.2))
    Vd = np.array([0, 5, 10, 15, 20.0])
    pd = np.array([1, 0.55, 0.33, 0.20, 0.11])
    Vg = np.linspace(0, 22, 300)
    ax = axs[0]
    ax.fill_between(Vg, p_of_V(Vg, beta=0.1098 * 1.2), p_of_V(Vg, beta=0.1098 * 0.8), color=COL["band"], lw=0)
    ax.plot(Vg, p_of_V(Vg), color=COL["edge"])
    ax.plot(Vd, pd, "o", color="k", ms=3.5)
    ax.set_xlabel("hornet load $V$")
    ax.set_ylabel("activity $p$")
    ax.set_title(r"(a)  $p=e^{-\beta V}$", loc="left", fontsize=9)
    ax = axs[1]
    ph = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    hf = np.array([0.202, 0.110, 0.060, 0.032, 0.018, 0.010])
    pg = np.linspace(0, 1, 300)
    ax.semilogy(pg, 100 * HF_of_p(pg), color=COL["edge"])
    ax.semilogy(ph, hf, "o", color="k", ms=3.5)
    ax.axvspan(0.5, 1, color="0.93", lw=0)
    ax.text(0.55, 0.1, "extra-\npolated", fontsize=7)
    ax.set_xlabel("activity $p$")
    ax.set_ylabel("homing failure [%]")
    ax.set_title("(b)  HF per return flight", loc="left", fontsize=9)
    ax = axs[2]
    ax.plot(Vg, 100 * mu_of_V(Vg) / P.m0, color=COL["traj"])
    ax.axhline(100 * mu_bound() / P.m0, ls="--", color="k", lw=0.8)
    ax.text(0.5, 100 * mu_bound() / P.m0 * 1.03, r"$\nu h_0/(\zeta e)$", fontsize=7)
    ax.set_xlabel("hornet load $V$")
    ax.set_ylabel(r"$\mu/m_0$ [%]")
    ax.set_ylim(0, 0.3)
    ax.set_title(r"(c)  $\mu=\nu p\,\mathrm{HF}(p)$", loc="left", fontsize=9)
    fig.tight_layout()
    return _save(fig, outdir, "calibration")


def fig_ladder(outdir, P=PAR):
    """Fig. 1.1: the three hornet loads that organise the results, with the theta bracket."""
    plt = _mpl()
    m0 = P.m0
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.55))
    th0 = [float(V_of_p(T(m0))) for T in (p_acc, p_star, p_E)]
    th1 = []
    for k in ("acc", "star", "E"):
        th1.append(float(V_of_p(threshold_with_m_of_p(k, lambda p: m0 * p + float(mu_of_p(p)), P))))
    labels = ["stores stop growing", "equilibrium lost", "collapse proved"]
    cols = [COL["f"], COL["edge"], COL["bound"]]
    ax.axvspan(0, 20, color="0.94", lw=0)
    ax.text(19.8, 3.55, "observed loads (Requier et al.)", ha="right", fontsize=7, color="0.4")
    ax.axvspan(5.5, 6.8, ymin=0.0, ymax=0.18, color=COL["f"], alpha=0.35, lw=0)
    ax.text(6.9, 0.25, "season store balance (numerical)", fontsize=7)
    for i, (a, b, lab, cc) in enumerate(zip(th0, th1, labels, cols)):
        y = 2.9 - 0.85 * i
        ax.plot([a, b], [y, y], color=cc, lw=5, alpha=0.35, solid_capstyle="butt")
        ax.plot(a, y, "o", color=cc, ms=6)
        ax.plot(b, y, "|", color=cc, ms=10, mew=2)
        ax.text(a - 0.25, y, f"{lab}  {a:.2f}", ha="right", va="center", fontsize=7.5)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 3.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"hornet load $V$   (dot: $\theta=0$, bar end: $\theta=1$)")
    fig.tight_layout()
    return _save(fig, outdir, "ladder")


def fig_schema_bio(outdir, P=PAR):
    """Fig. 3.1: compartment diagram of the published model, after Khoury et al. (2013), Fig. 1."""
    plt = _mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

    gold, red = "#b8860b", COL["bound"]
    dashdot = (0, (5, 2, 1, 2))
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.5))
    ax.set_xlim(0.1, 12.1)
    ax.set_ylim(0.45, 4.9)
    ax.axis("off")

    def box(x, y, w, h, title, ls="-", fc="white"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, lw=0.9, ls=ls, ec="k", fc=fc,
                                    boxstyle="round,pad=0.02,rounding_size=0.1"))
        ax.text(x, y, title, ha="center", va="center", fontsize=9)

    def arrow(a, b, color="k", ls="-", rad=0.0, lw=1.1):
        if ls == "-":
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                                         connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
            return
        # dashed or dotted shaft with a solid head, aligned with the tangent at the end point
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-", color=color, lw=lw, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        ctrl = (a + b) / 2 + rad * np.array([d[1], -d[0]])      # control point of arc3
        t = (b - ctrl) / np.linalg.norm(b - ctrl)
        ax.add_patch(FancyArrowPatch(tuple(b - 0.12 * t), tuple(b), arrowstyle="-|>", mutation_scale=9,
                                     color=color, lw=lw, shrinkA=0, shrinkB=0))

    def label(x, y, s, color="k", fs=7.4):
        ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color)

    box(1.0, 3.55, 1.5, 0.85, "uncapped\nbrood")
    box(3.35, 3.55, 1.4, 0.85, "capped\nbrood", ls="--", fc="#f5f5f5")
    box(5.95, 3.55, 1.6, 0.85, "hive bees")
    box(9.25, 3.55, 1.6, 0.85, "foragers")
    ax.add_patch(Ellipse((5.6, 1.0), 1.9, 0.8, fc="#fbf0d0", ec="k", lw=0.9))
    label(5.6, 1.0, "food stores", fs=9)
    box(10.95, 1.15, 1.9, 0.8, "Asian hornets", fc="#f6d5d1")

    arrow((1.75, 3.5), (2.65, 3.5))
    label(2.2, 3.28, "pupation", fs=7)
    arrow((4.05, 3.55), (5.15, 3.55))
    label(4.6, 3.28, "eclosion", fs=7)
    arrow((6.75, 3.72), (8.45, 3.72))
    label(7.6, 4.32, "transition to foraging", fs=7.2)
    label(7.6, 4.1, "(faster when food is short)", fs=6.5, color="0.3")
    arrow((8.45, 3.38), (6.75, 3.38), ls="--")
    label(7.7, 3.1, "social\ninhibition", fs=7.2)
    arrow((10.05, 3.55), (11.45, 3.55))
    label(10.75, 3.77, "death", fs=7.2)

    arrow((5.95, 4.0), (1.0, 4.0), ls=":", rad=0.22)
    label(3.45, 4.6, "hive bees nurse the brood", fs=7.2)
    arrow((4.65, 0.95), (0.8, 3.12), ls=":", rad=-0.1)
    label(1.3, 1.85, "food supply\naffects brood\nsurvival", fs=7)
    arrow((6.2, 1.3), (7.3, 3.6), ls=":")

    arrow((4.75, 1.25), (1.35, 3.12), ls=dashdot, lw=0.9)
    arrow((5.55, 1.4), (5.85, 3.12), ls=dashdot, lw=0.9)
    arrow((6.45, 1.2), (8.7, 3.12), ls=dashdot, lw=0.9)
    label(3.45, 2.5, "consumption", fs=7.2)

    arrow((9.2, 3.1), (6.55, 0.95), color=gold, lw=1.3)
    label(9.3, 2.3, "food collection\nby foragers", color=gold, fs=7.2)

    arrow((9.9, 3.1), (10.8, 1.56), color=red, lw=1.4)
    label(11.0, 2.5, "homing\nfailure", color=red)
    arrow((9.95, 1.05), (8.05, 1.97), color=red, ls=":", lw=1.4)
    label(8.75, 1.0, "foraging paralysis", color=red)

    handles = [Line2D([], [], color="k", lw=1.1), Line2D([], [], color="k", lw=1.1, ls="--"),
               Line2D([], [], color="k", lw=1.1, ls=":"), Line2D([], [], color="k", lw=0.9, ls=dashdot),
               Line2D([], [], color=gold, lw=1.3), Line2D([], [], color=red, lw=1.4)]
    labels = ["development and death", "social inhibition", "effect on a rate", "food consumption",
              "food collection", "hornet impacts"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7, handlelength=2.4,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.13, left=0, right=1, top=1)
    return _save(fig, outdir, "schema_bio")


def fig_axis3d(outdir, P=PAR):
    """Fig. 5.3: the collision at the existence edge in blow-up coordinates (Remark 5.28).

    Coordinates: forager fraction u = F/N, size N^(1/3), stores f. The food axis H = F = 0
    becomes the curve Gamma = {N = 0, u = u*(f)}, attracting below f_crit and repelling above.
    """
    plt = _mpl()
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection on older matplotlib)

    m = P.m0
    red, blue = COL["bound"], COL["edge"]
    fc = f_crit(m, P)
    f_max = 1500.0
    p_max = brentq(lambda p: equilibrium(p, m, P)[2] - f_max, p_star(m, P) + 1e-9, p_acc(m, P) - 1e-6)
    p_branch = np.linspace(p_star(m, P) + 1e-10, p_max, 600)
    X = np.array([equilibrium(p, m, P) for p in p_branch])
    Nb, ub = X[:, 0] + X[:, 1], X[:, 1] / (X[:, 0] + X[:, 1])
    fg = np.linspace(0, f_max, 400)
    ug = u_star(fg, m, P)
    p_traj = 0.45
    x_eq = equilibrium(p_traj, m, P)

    fig = plt.figure(figsize=(0.72 * TEXTWIDTH, 3.7))
    ax = fig.add_axes([-0.05, 0.03, 0.95, 1.0], projection="3d", computed_zorder=False)
    below = fg <= fc
    ax.plot(ug[below], 0 * fg[below], fg[below], color=red, lw=3.5, solid_capstyle="butt", zorder=5)
    ax.plot(ug[~below], 0 * fg[~below], fg[~below], color=blue, lw=3.5, solid_capstyle="butt", zorder=5)
    starts = [(60, 25, 200), (300, 120, 120), (1500, 500, 60), (40, 15, 290), (120, 30, 150), (30, 25, 250),
              (400, 150, 250), (2500, 1000, 900), (20000, 8000, 800), (200, 80, 1300), (15000, 6000, 1400),
              (8000, 1000, 300), (30000, 3000, 500)]
    for x0 in starts:
        sol = simulate_M(p_traj, m, x0, 3000.0, ration=True, P=P)
        H, F, f = sol["x"]
        N = H + F
        col = red if N[-1] < 1.0 else blue
        ok = N > 1e-6
        ax.plot(F[ok] / N[ok], np.cbrt(N[ok]), f[ok], color=col, lw=0.9, alpha=0.8, zorder=3)
        ax.scatter(x0[1] / (x0[0] + x0[1]), np.cbrt(x0[0] + x0[1]), x0[2], color=col, s=6, zorder=4)
    ax.plot(ub, np.cbrt(Nb), X[:, 2], color="k", lw=2.2, zorder=6)
    u_c = u_star(fc, m, P)
    ax.scatter([u_c], [0], [fc], color="k", s=28, zorder=7)
    ax.scatter(x_eq[1] / (x_eq[0] + x_eq[1]), np.cbrt(x_eq[0] + x_eq[1]), x_eq[2], marker="*",
               color=COL["f"], edgecolor="k", s=150, zorder=8)
    ax.text(u_c + 0.02, -5, fc - 50, r"$f_{\rm crit}$", fontsize=8.5, zorder=9)
    ax.text(0.55, 1.8, 170, r"$\Gamma$ attracts", color=red, fontsize=8, zorder=9)
    ax.text(0.27, 0, 1480, r"$\Gamma$ repels", color=blue, fontsize=8, zorder=9)
    ax.text(ub[-1], np.cbrt(Nb[-1]) + 1, X[-1, 2] + 30, "positive equilibria", fontsize=8, zorder=9)
    ax.set_xlabel("forager fraction $u=F/N$", labelpad=2)
    ax.set_ylabel("colony size $N^{1/3}$", labelpad=-2)
    fig.text(0.925, 0.55, "stores $f$ [g]", rotation=90, ha="center", va="center", fontsize=9)
    ax.set_xlim(0.15, 0.6)
    ax.set_ylim(0, 36)
    ax.set_zlim(0, f_max)
    ax.set_box_aspect((1.0, 1.35, 0.9))
    ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
    ax.tick_params(pad=-1, labelsize=7)
    ax.view_init(elev=17, azim=-57)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1)
    # a tight bounding box would cut off the 3d axis labels
    return _save(fig, outdir, "axis3d", tight=False)


def fig_basins(outdir, P=PAR):
    """Fig. 5.4: basins of attraction inside the existence band (Remark 5.30), about one minute.

    For each initial size N0 (ratio H:F = 2:1) a bisection in the initial stores f0 locates the
    boundary between collapse (H + F < 1 within 4000 d, with rationing) and the positive equilibrium.
    """
    plt = _mpl()
    m = P.m0
    u0 = 1.0 / 3.0
    fc = f_crit(m, P)

    def survives(p, N0, f0):
        sol = simulate_M(p, m, ((1 - u0) * N0, u0 * N0, f0), 4000.0, rtol=1e-8, atol=1e-8,
                         ration=True, stopN=1.0, P=P)
        return sol["stopped"] is None

    def boundary(p, N0, f_high=1200.0, steps=16):
        if not survives(p, N0, f_high):
            return np.nan
        if survives(p, N0, 1.0):
            return 0.0
        lo, hi = 1.0, f_high
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if survives(p, N0, mid) else (mid, hi)
        return 0.5 * (lo + hi)

    N_ax = np.logspace(0, 5, 90)
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.55), sharey=True)
    for ax, p, tag in zip(axs, (0.43, 0.48), "ab"):
        g = np.array([boundary(p, N0) for N0 in N_ax])
        ax.fill_between(N_ax, 0, g, color=COL["collapse"], lw=0)
        ax.fill_between(N_ax, g, 1200, color=COL["band"], lw=0)
        ax.plot(N_ax, g, color="k", lw=1)
        ax.axhline(fc, color="k", ls="--", lw=0.8)
        xe = equilibrium(p, m, P)
        ax.plot(xe[0] + xe[1], xe[2], "*", color=COL["f"], mec="k", ms=10)
        ax.set_xscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_ylim(0, 1200)
        ax.set_xlabel("initial adults $N_0$")
        ax.set_title(f"({tag})  $p={p}$,  $V={float(V_of_p(p, P)):.1f}$", loc="left", fontsize=9)
        ax.text(1.4, fc + 30, r"$f_{\rm crit}$", fontsize=8)
        ax.text(40, 1080, "positive equilibria", ha="center", fontsize=7.5)
        ax.text(1.4, 60, "collapse", fontsize=7.5)
    axs[0].set_ylabel("initial stores $f_0$ [g]")
    fig.tight_layout()
    return _save(fig, outdir, "basins")


# thesis figures in order of appearance; file name = key
FIGURES = dict(ladder=fig_ladder, schema_bio=fig_schema_bio, calibration=fig_calibration,
               endstates=fig_endstates, band=fig_band, axis3d=fig_axis3d, basins=fig_basins, pm=fig_pm)


def _save(fig, outdir, name, tight=True):
    """Save as PDF (for LaTeX) and PNG (for the web) and return the name."""
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    kw = {} if tight else {"bbox_inches": None}
    fig.savefig(os.path.join(outdir, name + ".pdf"), **kw)
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200, **kw)
    plt.close(fig)
    return name


# =============================================================================
# 10. MAIN
# =============================================================================

SECTIONS = dict(parameters=sec_parameters, khoury=sec_khoury, cascade=sec_cascade,
                random=sec_random_parameters, stability=sec_stability, globalb=sec_global,
                foodaxis=sec_food_axis, channels=sec_channels, sensitivity=sec_sensitivity,
                season=sec_season, delay=sec_delay)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the finite-horizon threshold search (Tab. B.3, ~1 min)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--figs", default="all", help="comma-separated figure names or 'all'")
    ap.add_argument("--sections", default="all", help="comma-separated section names or 'all'")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("sections:", ", ".join(SECTIONS))
        print("figures: ", ", ".join(FIGURES))
        return 0
    t0 = time.time()
    rep = Report()
    secs = list(SECTIONS) if a.sections == "all" else [s.strip() for s in a.sections.split(",")]
    for s in secs:
        ts = time.time()
        SECTIONS[s](rep, PAR, full=not a.quick)
        print(f"  ... {s}: {time.time() - ts:.1f} s")
    os.makedirs(a.outdir, exist_ok=True)
    rep.write(os.path.join(a.outdir, "verification_report.txt"), os.path.join(a.outdir, "verification_report.csv"))
    if not a.no_figs:
        figs = list(FIGURES) if a.figs == "all" else [f.strip() for f in a.figs.split(",")]
        print("\nfigures:")
        for fname in figs:
            ts = time.time()
            FIGURES[fname](os.path.join(a.outdir, "figs"), PAR)
            print(f"  {fname:<12} {time.time() - ts:.1f} s")
    n, cnt = rep.summary()
    print(f"\n{n} checks: {cnt['PASS']} PASS, {cnt['DIFF']} DIFF, {cnt['FAIL']} FAIL, {cnt['INFO']} INFO "
          f"({time.time() - t0:.0f} s)")
    return 0 if cnt["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
