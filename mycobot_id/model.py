from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def simulate_second_order(
    t: np.ndarray,
    u: np.ndarray,
    omega_n: float,
    zeta: float,
    delay_s: float,
    gain: float,
    y0: float,
) -> np.ndarray:
    """Simulate G(s)=gain*wn^2/(s^2+2*zeta*wn*s+wn^2)e^(-Ls)."""
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("t must contain at least two strictly increasing timestamps")
    input_baseline = float(u[0])
    def delayed_command(current_t: float) -> float:
        # Robot position commands are held until the next command (zero-order
        # hold). Linear interpolation would invent a ramp before a sampled step
        # and bias the fitted delay downward.
        query_t = current_t - delay_s
        index = int(np.searchsorted(t, query_t, side="right") - 1)
        index = min(max(index, 0), len(u) - 1)
        return float(u[index] - input_baseline)

    def dynamics(current_t: float, state: np.ndarray) -> list[float]:
        command = delayed_command(current_t)
        position_delta, velocity = state
        acceleration = omega_n**2 * (gain * command - position_delta)
        acceleration -= 2.0 * zeta * omega_n * velocity
        return [velocity, acceleration]

    result = solve_ivp(
        dynamics, (float(t[0]), float(t[-1])), [0.0, 0.0],
        t_eval=t, rtol=1e-6, atol=1e-8,
        max_step=float(np.min(np.diff(t))) / 2.0,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(y0) + result.y[0]
