"""Analytic unit-step candidates for closed-loop position identification."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def predict(model, parameters, t, delta, y0):
    t = np.asarray(t, dtype=float)
    if model == "second_order":
        wn, zeta, delay, gain = parameters
        elapsed = np.maximum(t - delay, 0.0)
        if zeta < 1.0 - 1e-5:
            factor = np.sqrt(1.0 - zeta * zeta)
            response = 1.0 - np.exp(-zeta * wn * elapsed) * (
                np.cos(wn * factor * elapsed) + zeta / factor * np.sin(wn * factor * elapsed))
        elif zeta > 1.0 + 1e-5:
            factor = np.sqrt(zeta * zeta - 1.0)
            r1, r2 = -wn * (zeta - factor), -wn * (zeta + factor)
            response = 1.0 + (r2 * np.exp(r1 * elapsed) - r1 * np.exp(r2 * elapsed)) / (r1 - r2)
        else:
            response = 1.0 - np.exp(-wn * elapsed) * (1.0 + wn * elapsed)
        return y0 + gain * delta * response
    if model == "first_order":
        tau, delay, gain = parameters
        elapsed = np.maximum(t - delay, 0.0)
        return y0 + gain * delta * (1.0 - np.exp(-elapsed / tau))
    if model in ("velocity_limited", "deadzone_velocity"):
        if model == "deadzone_velocity":
            rate_gain, vmax, delay, deadband = parameters
            gain = 1.0
        else:
            rate_gain, vmax, delay, gain = parameters
            deadband = 0.0
        elapsed = np.maximum(t - delay, 0.0)
        amplitude = max(abs(gain * delta) - deadband, 0.0)
        threshold = vmax / rate_gain
        if amplitude <= threshold:
            distance = amplitude * (1.0 - np.exp(-rate_gain * elapsed))
        else:
            transition = (amplitude - threshold) / vmax
            tail = amplitude - threshold * np.exp(-rate_gain * np.maximum(elapsed - transition, 0.0))
            distance = np.where(elapsed <= transition, vmax * elapsed, tail)
        return y0 + np.sign(delta) * distance
    raise ValueError(model)


BOUNDS = {
    "second_order": ([0.2, 0.05, 0.0, 0.2], [150.0, 5.0, 0.5, 1.8]),
    "first_order": ([0.003, 0.0, 0.2], [5.0, 0.5, 1.8]),
    "velocity_limited": ([0.2, 0.1, 0.0, 0.2], [200.0, 100.0, 0.5, 1.8]),
    "deadzone_velocity": ([0.2, 0.1, 0.0, 0.0], [200.0, 100.0, 0.5, 2.0]),
}
STARTS = {
    "second_order": [[5.0, 0.8, 0.03, 1.0], [20.0, 1.2, 0.05, 1.0], [60.0, 0.8, 0.02, 1.0]],
    "first_order": [[0.1, 0.03, 1.0], [0.4, 0.05, 1.0]],
    "velocity_limited": [[10.0, 10.0, 0.03, 1.0], [30.0, 30.0, 0.05, 1.0], [5.0, 5.0, 0.02, 1.0]],
    "deadzone_velocity": [[10.0, 10.0, 0.05, 0.8], [30.0, 30.0, 0.05, 1.0], [5.0, 5.0, 0.02, 0.5]],
}
NAMES = {
    "second_order": ["omega_n_rad_s", "zeta", "delay_s", "dc_gain"],
    "first_order": ["tau_s", "delay_s", "dc_gain"],
    "velocity_limited": ["rate_gain_per_s", "vmax_deg_s", "delay_s", "dc_gain"],
    "deadzone_velocity": ["rate_gain_per_s", "vmax_deg_s", "delay_s", "deadband_deg"],
}


def errors(model, parameters, events):
    return np.concatenate([
        (predict(model, parameters, event["t"], event["error"] if model == "deadzone_velocity" else event["delta"], event["y0"]) - event["y"])[event["t"] >= 0]
        for event in events])


def fit_model(model, events):
    if not events or any(abs(event["delta"]) < 0.25 for event in events):
        raise ValueError("Missing input excitation")
    fits = [least_squares(lambda p: errors(model, p, events), start, bounds=BOUNDS[model],
                          x_scale="jac", max_nfev=400, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            for start in STARTS[model]]
    best = min(fits, key=lambda fit: float(np.sum(fit.fun**2)))
    singular = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
    lower, upper = map(np.asarray, BOUNDS[model])
    near = ((best.x - lower) / (upper - lower) < 0.001) | ((upper - best.x) / (upper - lower) < 0.001)
    return {
        "parameters": dict(zip(NAMES[model], map(float, best.x))),
        "parameter_vector": list(map(float, best.x)), "success": bool(best.success),
        "message": str(best.message), "jacobian_condition_number": condition,
        "near_bound_parameters": [name for name, flag in zip(NAMES[model], near) if flag],
        "train_rmse_deg": float(np.sqrt(np.mean(best.fun**2))),
    }
