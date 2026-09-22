"""Non-iterative action estimator for the two-field quartic potential.

This module implements the prescription in the associated 2FAST paper.  It is
deliberately limited to

    V(h, s) = -a_h h^2/2 - a_s s^2/2
              + lambda_h h^4/4 + lambda_s s^4/4
              + lambda_hs h^2 s^2/4.

The public entry points distinguish the physical/metastability checks, the
empirically validated coefficient domain, and failures of the analytic profile
construction.  An action is returned only if all three stages pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, pi, sqrt
from typing import Any, Callable

import numpy as np
from numpy.polynomial import Polynomial


@dataclass(frozen=True)
class QuarticParameters:
    """Five independent coefficients of the two-field quartic potential."""

    a_h: float
    a_s: float
    lambda_h: float
    lambda_s: float
    lambda_hs: float


@dataclass(frozen=True)
class SSMParameters:
    """Inputs of the leading-thermal-mass Z2 singlet model used in the paper."""

    m_s: float
    lambda_s: float
    lambda_hs: float
    temperature: float
    v_0: float = 246.0
    m_h: float = 125.1
    g: float = 0.653
    g_prime: float = 0.358
    y_t: float = 0.995


@dataclass(frozen=True)
class FailedCondition:
    """One failed input, physics, validation-domain, or construction check."""

    code: str
    message: str
    value: float | None = None
    requirement: str | None = None
    stage: str = "construction"


@dataclass(frozen=True)
class ActionEstimate:
    """Result returned by the quartic estimator."""

    valid: bool
    action: float | None
    s3_over_t: float | None
    branch: str | None
    failures: tuple[FailedCondition, ...]
    shape: dict[str, float]
    coefficients: QuarticParameters | None
    source: str
    diagnostics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ConstructionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        value: float | None = None,
        requirement: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = FailedCondition(code, message, value, requirement)


@dataclass(frozen=True)
class _Potential:
    a_h: float
    a_s: float
    lambda_h: float
    lambda_s: float
    lambda_hs: float

    @property
    def v(self) -> float:
        return sqrt(self.a_h / self.lambda_h)

    @property
    def w(self) -> float:
        return sqrt(self.a_s / self.lambda_s)

    @property
    def v_false(self) -> float:
        return -(self.a_s**2) / (4.0 * self.lambda_s)

    def value(self, h: np.ndarray, s: np.ndarray) -> np.ndarray:
        return (
            -0.5 * self.a_h * h**2
            - 0.5 * self.a_s * s**2
            + 0.25 * self.lambda_h * h**4
            + 0.25 * self.lambda_s * s**4
            + 0.25 * self.lambda_hs * h**2 * s**2
        )

    def shifted(self, h: np.ndarray, s: np.ndarray) -> np.ndarray:
        return self.value(h, s) - self.v_false

    def gradient(self, h: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            h * (-self.a_h + self.lambda_h * h**2 + 0.5 * self.lambda_hs * s**2),
            s * (-self.a_s + self.lambda_s * s**2 + 0.5 * self.lambda_hs * h**2),
        )

    def hessian(
        self, h: np.ndarray, s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            -self.a_h + 3.0 * self.lambda_h * h**2 + 0.5 * self.lambda_hs * s**2,
            self.lambda_hs * h * s,
            -self.a_s + 3.0 * self.lambda_s * s**2 + 0.5 * self.lambda_hs * h**2,
        )


def _finite_positive(name: str, value: float, failures: list[FailedCondition]) -> None:
    if not np.isfinite(value) or value <= 0.0:
        failures.append(
            FailedCondition(
                f"invalid_{name}",
                f"{name} 必须是有限正数",
                float(value),
                f"{name} > 0",
                "input",
            )
        )


def _shape_variables(parameters: QuarticParameters) -> dict[str, float]:
    gamma = sqrt(parameters.lambda_h / parameters.lambda_s)
    eta = parameters.lambda_hs / (
        2.0 * sqrt(parameters.lambda_h * parameters.lambda_s)
    ) - 1.0
    d = parameters.a_h / (gamma * parameters.a_s) - 1.0
    epsilon = d / eta if eta != 0.0 else np.nan
    return {"gamma": gamma, "eta": eta, "d": d, "epsilon": epsilon}


def _validate_quartic(
    parameters: QuarticParameters,
) -> tuple[dict[str, float], str | None, list[FailedCondition]]:
    failures: list[FailedCondition] = []
    for name, value in asdict(parameters).items():
        _finite_positive(name, value, failures)
    if failures:
        return {}, None, failures

    shape = _shape_variables(parameters)
    gamma, eta, d, epsilon = (
        shape["gamma"],
        shape["eta"],
        shape["d"],
        shape["epsilon"],
    )

    if eta <= 0.0:
        failures.append(
            FailedCondition(
                "mixed_barrier",
                "混合耦合不足以使两个轴上极小值由势垒分开",
                eta,
                "eta > 0",
                "physics",
            )
        )
    if d <= 0.0:
        failures.append(
            FailedCondition(
                "vacuum_ordering",
                "(v,0) 不是比 (0,w) 更深的真空",
                d,
                "d > 0",
                "physics",
            )
        )
    if np.isfinite(epsilon) and epsilon >= 1.0:
        failures.append(
            FailedCondition(
                "false_vacuum_metastability",
                "(0,w) 在 h 方向已失稳，不再是所需的亚稳假真空",
                epsilon,
                "epsilon < 1",
                "physics",
            )
        )
    if np.isfinite(epsilon) and epsilon <= 0.0 and d > 0.0:
        failures.append(
            FailedCondition(
                "metastable_interval",
                "参数不在论文研究的亚稳区间",
                epsilon,
                "epsilon > 0",
                "physics",
            )
        )
    if failures:
        return shape, None, failures

    # Equation (domain) of the paper.  This is an empirical reliability domain,
    # not a physical phase boundary.
    branch: str | None = None
    if eta >= 1.0 and epsilon < 0.4:
        branch = "conic"
        if d >= 1.0:
            failures.append(
                FailedCondition(
                    "validated_depth_domain",
                    "强混合势垒分支的真空深度差超出数值验证域",
                    d,
                    "d < 1",
                    "validation_domain",
                )
            )
    elif 0.0 < eta < 1.0:
        branch = "moment"
        if not (0.5 < gamma < 2.0):
            failures.append(
                FailedCondition(
                    "validated_gradient_domain",
                    "两场的梯度代价相差过大，超出数值验证域",
                    gamma,
                    "0.5 < gamma < 2",
                    "validation_domain",
                )
            )
    else:
        failures.append(
            FailedCondition(
                "validated_branch_domain",
                "该强势垒且接近失稳的区域未获得论文的精度验证",
                epsilon,
                "eta >= 1 时需 epsilon < 0.4",
                "validation_domain",
            )
        )
    return shape, branch, failures


def _dimensionless_potential(shape: dict[str, float]) -> _Potential:
    gamma, eta, d = shape["gamma"], shape["eta"], shape["d"]
    return _Potential(
        a_h=gamma * (1.0 + d),
        a_s=1.0,
        lambda_h=gamma**2,
        lambda_s=1.0,
        lambda_hs=2.0 * gamma * (1.0 + eta),
    )


def _log_cosh(x: np.ndarray | float) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    absolute = np.abs(x_array)
    return absolute + np.log1p(np.exp(-2.0 * absolute)) - log(2.0)


def _log_sinh_positive(x: float) -> float:
    if x <= 0.0:
        return -np.inf
    if x < 20.0:
        return log(np.sinh(x))
    return x + np.log1p(-np.exp(-2.0 * x)) - log(2.0)


def _b_from_alpha(alpha: float) -> float:
    if alpha < 350.0:
        return 2.0 * np.arcsinh(np.exp(alpha))
    return 2.0 * (alpha + log(2.0))


def _sin_cos_theta(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f, dtype=float)
    positive = f >= 0.0
    small = np.exp(-np.abs(f))
    denominator = np.sqrt(1.0 + small**2)
    sin_theta = np.where(positive, 1.0 / denominator, small / denominator)
    cos_theta = np.where(positive, small / denominator, 1.0 / denominator)
    return sin_theta, cos_theta


def _gauss_integral(
    function: Callable[[np.ndarray], np.ndarray],
    boundaries: list[float],
    order: int = 64,
) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    clean = sorted({max(0.0, float(item)) for item in boundaries if np.isfinite(item)})
    clean = [item for i, item in enumerate(clean) if i == 0 or item > clean[i - 1]]
    total = 0.0
    for left, right in zip(clean[:-1], clean[1:], strict=True):
        if right <= left:
            continue
        radius = 0.5 * (right - left) * nodes + 0.5 * (right + left)
        values = np.asarray(function(radius), dtype=float)
        if np.any(~np.isfinite(values)):
            raise _ConstructionError(
                "nonfinite_integrand", "最终作用量积分中出现非有限值"
            )
        total += 0.5 * (right - left) * float(np.dot(weights, values))
    return total


def _normal_quantities(
    potential: _Potential,
    h: np.ndarray,
    s: np.ndarray,
    n_h: np.ndarray,
    n_s: np.ndarray,
    kappa: np.ndarray,
    ell_prime_sq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grad_h, grad_s = potential.gradient(h, s)
    h_h, h_s, s_s = potential.hessian(h, s)
    v_n = n_h * grad_h + n_s * grad_s
    v_nn = n_h**2 * h_h + 2.0 * n_h * n_s * h_s + n_s**2 * s_s
    residual = v_n - kappa * ell_prime_sq
    restoring = v_nn + kappa**2 * ell_prime_sq
    return residual, restoring


def _conic_action(
    potential: _Potential, shape: dict[str, float]
) -> tuple[float, dict[str, float]]:
    v, w = potential.v, potential.w
    delta = potential.v_false - potential.value(np.asarray(v), np.asarray(0.0))
    f_sum = v**2 + w**2
    e_sum = potential.lambda_h * v**2 + potential.lambda_s * w**2
    c_sum = potential.lambda_h * v**4 + potential.lambda_s * w**4
    p_mix = potential.lambda_hs * v**2 * w**2

    coefficients = np.array(
        [
            2.0 * c_sum,
            2.0 * f_sum * e_sum - 2.0 * c_sum,
            p_mix
            - c_sum
            + 3.0 * f_sum * e_sum
            - 0.5 * potential.lambda_hs * f_sum**2,
            c_sum
            - p_mix
            + f_sum * e_sum
            - 0.5 * potential.lambda_hs * f_sum**2,
        ]
    )
    roots = np.roots(coefficients)
    nonnegative = [
        float(root.real)
        for root in roots
        if abs(root.imag) <= 1.0e-8 * (1.0 + abs(root.real)) and root.real >= 0.0
    ]
    if len(nonnegative) != 1:
        raise _ConstructionError(
            "conic_root",
            "中点法向平衡方程没有唯一的非负实根",
            value=float(len(nonnegative)),
            requirement="恰有一个 c >= 0 的实根",
        )
    c = nonnegative[0]
    w_mid = (c_sum * (2.0 * c**2 - 1.0) + p_mix) / (
        16.0 * (1.0 + c) ** 2
    )
    if w_mid <= 0.0 or not np.isfinite(w_mid):
        raise _ConstructionError(
            "conic_barrier",
            "辅助势垒的中点高度不是正数",
            value=w_mid,
            requirement="W_m > 0",
        )

    h_mid = (v**2 + w**2) / 4.0
    length = sqrt(h_mid / (2.0 * w_mid))
    tension = (4.0 / 3.0) * sqrt(2.0 * w_mid * h_mid)
    radius = 2.0 * tension / delta
    alpha = float(_log_cosh(radius / length))

    def geometry(d_value: np.ndarray) -> tuple[np.ndarray, ...]:
        q = np.sqrt((2.0 + (c - 1.0) * d_value**2) / (1.0 + c))
        q_d = (c - 1.0) * d_value / ((1.0 + c) * q)
        q_dd = (c - 1.0) / ((1.0 + c) * q) - (
            (c - 1.0) ** 2 * d_value**2 / ((1.0 + c) ** 2 * q**3)
        )
        h_d = 0.5 * v * (q_d + 1.0)
        s_d = 0.5 * w * (q_d - 1.0)
        h_dd = 0.5 * v * q_dd
        s_dd = 0.5 * w * q_dd
        metric = h_d**2 + s_d**2
        root_metric = np.sqrt(metric)
        h = 0.5 * v * (q + d_value)
        s = 0.5 * w * (q - d_value)
        n_h = -s_d / root_metric
        n_s = h_d / root_metric
        kappa = (h_d * s_dd - s_d * h_dd) / metric**1.5
        return h, s, metric, n_h, n_s, kappa

    zero = np.asarray(0.0)
    h0, s0, metric0, *_ = geometry(zero)
    grad_h0, grad_s0 = potential.gradient(h0, s0)
    q0 = sqrt(2.0 / (1.0 + c))
    h_d0, s_d0 = 0.5 * v, -0.5 * w
    ubar_d = grad_h0 * h_d0 + grad_s0 * s_d0 + 0.75 * delta
    metric_d0 = (v**2 - w**2) * (c - 1.0) / (2.0 * (1.0 + c) * q0)
    beta = 0.5 * (float(ubar_d) / w_mid - metric_d0 / float(metric0))
    if abs(beta) >= 0.95:
        raise _ConstructionError(
            "conic_asymmetry",
            "规则化圆锥壁的非对称参数超出允许范围",
            value=beta,
            requirement="|beta| < 0.95",
        )

    def integrand(radial: np.ndarray) -> np.ndarray:
        x = radial / length
        z = alpha - _log_cosh(x)
        f = z + beta * _log_cosh(z)
        d_value = np.tanh(f)
        z_prime = -np.tanh(x) / length
        f_prime = z_prime * (1.0 + beta * np.tanh(z))
        d_prime = (1.0 - d_value**2) * f_prime
        h, s, metric, n_h, n_s, kappa = geometry(d_value)
        ell_prime_sq = metric * d_prime**2
        residual, restoring = _normal_quantities(
            potential, h, s, n_h, n_s, kappa, ell_prime_sq
        )
        if np.any(restoring <= 0.0):
            raise _ConstructionError(
                "transverse_restoring_force",
                "局域法向修正遇到非正的横向恢复系数",
                value=float(np.min(restoring)),
                requirement="M_perp^2 > 0",
            )
        energy = (
            0.5 * ell_prime_sq
            + potential.shifted(h, s)
            - residual**2 / (2.0 * restoring)
        )
        return 4.0 * pi * radial**2 * energy

    tail = 24.0 * length / (1.0 - beta)
    boundaries = [
        0.0,
        max(0.0, radius - 8.0 * length),
        radius,
        radius + 8.0 * length,
        radius + tail,
    ]
    action = _gauss_integral(integrand, boundaries)
    if not np.isfinite(action) or action <= 0.0:
        raise _ConstructionError(
            "nonpositive_action",
            "圆锥分支没有得到有限正作用量",
            value=action,
            requirement="S_c > 0",
        )
    return action, {
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "L_dimensionless": length,
        "R_dimensionless": radius,
    }


def _ellipse_geometry(
    potential: _Potential, theta: np.ndarray
) -> tuple[np.ndarray, ...]:
    v, w = potential.v, potential.w
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    metric = v**2 * cos_theta**2 + w**2 * sin_theta**2
    root_metric = np.sqrt(metric)
    h, s = v * sin_theta, w * cos_theta
    t_h, t_s = v * cos_theta / root_metric, -w * sin_theta / root_metric
    n_h, n_s = w * sin_theta / root_metric, v * cos_theta / root_metric
    kappa = -v * w / metric**1.5
    return h, s, metric, t_h, t_s, n_h, n_s, kappa


def _moment_polynomials(t_value: complex) -> tuple[Polynomial, ...]:
    t = t_value
    p_poly = Polynomial([0.0, pi**2, 0.0, 1.0])
    q_poly = Polynomial([pi**2, 0.0, 3.0])
    b_poly = Polynomial([0.0, 1.0])
    m1 = t * p_poly / 24.0
    m2 = t * ((1.0 + t**2) * p_poly - 2.0 * t * q_poly) / 48.0
    m3 = t * (
        (3.0 + 2.0 * t**2 + 3.0 * t**4) * p_poly
        - 6.0 * t * (1.0 + t**2) * q_poly
        + 24.0 * t**2 * b_poly
    ) / 192.0
    m4 = t * (
        (15.0 + 9.0 * t**2 + 9.0 * t**4 + 15.0 * t**6) * p_poly
        - 2.0 * t * (15.0 + 14.0 * t**2 + 15.0 * t**4) * q_poly
        + 144.0 * t**2 * (1.0 + t**2) * b_poly
        - 48.0 * t**3
    ) / 1152.0
    return m1, m2, m3, m4


def _kw_polynomials(
    t_value: complex, g_coefficients: np.ndarray, u_coefficients: np.ndarray
) -> tuple[Polynomial, Polynomial]:
    moments = _moment_polynomials(t_value)
    c_t = t_value**-2
    g0, g1, g2 = g_coefficients
    kinetic = 0.5 * (
        g0 * moments[0]
        + (g1 - c_t * g0) * moments[1]
        + (g2 - c_t * g1) * moments[2]
        - c_t * g2 * moments[3]
    )
    potential = sum(
        (
            coefficient * moment
            for coefficient, moment in zip(u_coefficients, moments, strict=True)
        ),
        Polynomial([0.0]),
    )
    return kinetic, potential


def _polynomial_coefficients(poly: Polynomial, size: int = 4) -> np.ndarray:
    coefficients = np.zeros(size, dtype=complex)
    coefficients[: len(poly.coef)] = poly.coef
    return coefficients


def _moment_action(
    potential: _Potential, shape: dict[str, float]
) -> tuple[float, dict[str, float]]:
    gamma, eta, epsilon = shape["gamma"], shape["eta"], shape["epsilon"]
    v, w = potential.v, potential.w
    delta = potential.v_false - potential.value(np.asarray(v), np.asarray(0.0))
    lambda_barrier = (
        potential.lambda_hs * v**2 * w**2
        - potential.lambda_h * v**4
        - potential.lambda_s * w**4
    ) / 4.0
    if lambda_barrier <= 0.0:
        raise _ConstructionError(
            "ellipse_barrier",
            "椭圆辅助路径没有正势垒",
            value=lambda_barrier,
            requirement="Lambda > 0",
        )

    def relaxed(theta_value: float) -> np.ndarray:
        theta = np.asarray(theta_value)
        h, s, metric, _, _, n_h, n_s, kappa = _ellipse_geometry(potential, theta)
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        speed_sq = 2.0 * lambda_barrier * sin_theta**2 * cos_theta**2
        residual, restoring = _normal_quantities(
            potential, h, s, n_h, n_s, kappa, speed_sq
        )
        # F = V_n-kappa nu^2, hence zeta_nu = -F/M^2.
        if float(restoring) <= 0.0:
            raise _ConstructionError(
                "seed_restoring_force",
                "椭圆中点的辅助法向恢复系数不是正数",
                value=float(restoring),
                requirement="D_nu > 0",
            )
        zeta = -residual / restoring
        return np.array([h + zeta * n_h, s + zeta * n_s], dtype=float)

    theta_mid = pi / 4.0
    step = 2.0e-4
    psi_m2 = relaxed(theta_mid - 2.0 * step)
    psi_m1 = relaxed(theta_mid - step)
    psi_0 = relaxed(theta_mid)
    psi_p1 = relaxed(theta_mid + step)
    psi_p2 = relaxed(theta_mid + 2.0 * step)
    psi_theta = (psi_m2 - 8.0 * psi_m1 + 8.0 * psi_p1 - psi_p2) / (
        12.0 * step
    )
    psi_theta_theta = (
        -psi_p2 + 16.0 * psi_p1 - 30.0 * psi_0 + 16.0 * psi_m1 - psi_m2
    ) / (12.0 * step**2)
    h0 = float(np.dot(psi_theta, psi_theta))
    h0_prime = float(2.0 * np.dot(psi_theta, psi_theta_theta))
    w0 = float(potential.shifted(psi_0[0], psi_0[1]) + delta / 2.0)
    grad_h, grad_s = potential.gradient(psi_0[0], psi_0[1])
    w0_prime = float(grad_h * psi_theta[0] + grad_s * psi_theta[1] + delta)
    if h0 <= 0.0 or w0 <= 0.0:
        raise _ConstructionError(
            "wall_seed",
            "变量振幅分支的壁极限种子不是正定的",
            value=min(h0, w0),
            requirement="H_0 > 0 且 W_0 > 0",
        )

    length0_sq = h0 / (8.0 * w0)
    alpha0 = float(_log_cosh(8.0 * w0 / delta))
    beta0 = 0.25 * (w0_prime / w0 - h0_prime / h0)

    mass_h_false_sq = gamma * eta * (1.0 - epsilon)
    mass_s_false_sq = 2.0
    mass_s_true_sq = (1.0 + eta) * (1.0 + shape["d"]) - 1.0
    soft_g = potential.lambda_hs**2 / (4.0 * potential.lambda_s) - potential.lambda_h
    j2 = pi**2 / 12.0
    j4 = pi**2 / 18.0 - 1.0 / 3.0
    k2 = j2 - j4
    amplitude_soft_sq = 4.0 * mass_h_false_sq * j2 / (soft_g * j4)
    length_soft_sq = k2 / (3.0 * mass_h_false_sq * j2)
    weight = epsilon**4
    alpha_star = (1.0 - weight) * alpha0 + weight * log(
        sqrt(amplitude_soft_sq) / v
    )
    beta_star = (1.0 - weight) * beta0
    length_star_sq = (1.0 - weight) * length0_sq + weight * length_soft_sq
    if (
        not np.isfinite(alpha_star)
        or not np.isfinite(beta_star)
        or length_star_sq <= 0.0
    ):
        raise _ConstructionError(
            "moment_seed", "变量振幅分支的初始参数无效"
        )

    g_b = 4.0 * h0 - 2.0 * (v**2 + w**2)
    g_coefficients = np.array([v**2, w**2 - v**2 + g_b, -g_b])
    k_star = (1.0 - beta_star) / sqrt(length_star_sq)
    e_star = (k_star**2 + mass_h_false_sq + mass_s_true_sq) / mass_s_false_sq
    q_star = 0.5 + e_star
    a_false = 0.5 * mass_h_false_sq * v**2
    c_true = -0.5 * mass_s_true_sq * w**2
    a2 = (
        -mass_h_false_sq * w**2 * e_star
        + 0.5 * mass_s_false_sq * w**2 * q_star**2
        - 0.5 * potential.lambda_hs * w**2 * v**2 * q_star
        + 0.25 * potential.lambda_h * v**4
    )
    u_coefficients = np.array(
        [
            a_false,
            a2,
            -c_true - 3.0 * a_false - 2.0 * a2 - 4.0 * delta,
            c_true + 2.0 * a_false + a2 + 3.0 * delta,
        ]
    )

    b_star = _b_from_alpha(alpha_star)
    t_star = float(np.tanh(b_star / 2.0))
    kinetic, integrated_potential = _kw_polynomials(
        t_star, g_coefficients, u_coefficients
    )
    complex_step = 1.0e-20
    kinetic_complex, potential_complex = _kw_polynomials(
        t_star + 1j * complex_step, g_coefficients, u_coefficients
    )
    kinetic_t = Polynomial(
        np.imag(_polynomial_coefficients(kinetic_complex)) / complex_step
    )
    potential_t = Polynomial(
        np.imag(_polynomial_coefficients(potential_complex)) / complex_step
    )
    d_alpha_kinetic = (
        2.0 * t_star * kinetic.deriv()
        + t_star * (1.0 - t_star**2) * kinetic_t
    )
    d_alpha_potential = (
        2.0 * t_star * integrated_potential.deriv()
        + t_star * (1.0 - t_star**2) * potential_t
    )
    amplitude_polynomial = (
        3.0 * d_alpha_kinetic * integrated_potential
        - kinetic * d_alpha_potential
    )
    roots = amplitude_polynomial.roots()
    candidates: list[float] = []
    for root in roots:
        if abs(root.imag) > 1.0e-7 * (1.0 + abs(root.real)) or root.real <= 0.0:
            continue
        b_value = float(root.real)
        k_value = float(np.real(kinetic(b_value)))
        w_value = float(np.real(integrated_potential(b_value)))
        slope = float(np.real(amplitude_polynomial.deriv()(b_value)))
        if k_value > 0.0 and w_value < 0.0 and slope < 0.0:
            candidates.append(b_value)
    if not candidates:
        raise _ConstructionError(
            "moment_root", "振幅代数方程没有满足稳定性筛选的正实根"
        )
    b_value = min(candidates, key=lambda item: abs(item - b_star))
    alpha = _log_sinh_positive(b_value / 2.0)
    t_value = float(np.tanh(b_value / 2.0))
    kinetic_final, potential_final = _kw_polynomials(
        t_value, g_coefficients, u_coefficients
    )
    kinetic_value = float(np.real(kinetic_final(b_value)))
    potential_value = float(np.real(potential_final(b_value)))
    if kinetic_value <= 0.0 or potential_value >= 0.0:
        raise _ConstructionError(
            "moment_scale", "恢复物理振幅后不能得到正的径向尺度"
        )
    length_sq = -kinetic_value / (3.0 * potential_value)
    length = sqrt(length_sq)
    beta = beta_star
    if abs(beta) >= 0.85:
        raise _ConstructionError(
            "moment_asymmetry",
            "变量振幅分支的非对称参数超出允许范围",
            value=beta,
            requirement="|beta| < 0.85",
        )
    if not (-8.0 < alpha < 4000.0):
        raise _ConstructionError(
            "moment_amplitude_range",
            "变量振幅超出数值处方允许范围",
            value=alpha,
            requirement="-8 < alpha < 4000",
        )

    def reference(radial: np.ndarray) -> tuple[np.ndarray, ...]:
        x = radial / length
        z = alpha - _log_cosh(x)
        f = z + beta * _log_cosh(z)
        sin_theta, cos_theta = _sin_cos_theta(f)
        z_prime = -np.tanh(x) / length
        f_prime = z_prime * (1.0 + beta * np.tanh(z))
        theta_prime = sin_theta * cos_theta * f_prime
        h = v * sin_theta
        s = w * cos_theta
        metric = v**2 * cos_theta**2 + w**2 * sin_theta**2
        root_metric = np.sqrt(metric)
        n_h = w * sin_theta / root_metric
        n_s = v * cos_theta / root_metric
        kappa = -v * w / metric**1.5
        ell_prime_sq = metric * theta_prime**2
        residual, restoring = _normal_quantities(
            potential, h, s, n_h, n_s, kappa, ell_prime_sq
        )
        if np.any(restoring <= 0.0):
            raise _ConstructionError(
                "transverse_restoring_force",
                "局域法向修正遇到非正的横向恢复系数",
                value=float(np.min(restoring)),
                requirement="M_perp^2 > 0",
            )
        zeta = -residual / restoring
        return h, s, n_h, n_s, kappa, ell_prime_sq, zeta

    derivative_step = max(1.0e-6, 2.0e-5 * length)

    def integrand(radial: np.ndarray) -> np.ndarray:
        h, s, n_h, n_s, kappa, ell_prime_sq, zeta = reference(radial)
        zeta_plus = reference(radial + derivative_step)[-1]
        zeta_minus = reference(radial - derivative_step)[-1]
        zeta_prime = (zeta_plus - zeta_minus) / (2.0 * derivative_step)
        h_displaced = h + zeta * n_h
        s_displaced = s + zeta * n_s
        kinetic_density = 0.5 * (
            (1.0 - kappa * zeta) ** 2 * ell_prime_sq + zeta_prime**2
        )
        energy = kinetic_density + potential.shifted(h_displaced, s_displaced)
        return 4.0 * pi * radial**2 * energy

    if alpha > 0.0:
        if alpha < 350.0:
            wall_x = float(np.arccosh(np.exp(alpha)))
        else:
            wall_x = alpha + log(2.0)
        wall_radius = length * wall_x
        boundaries = [
            0.0,
            max(0.0, wall_radius - 8.0 * length),
            wall_radius,
            wall_radius + 8.0 * length,
            wall_radius + 24.0 * length / (1.0 - beta),
        ]
    else:
        boundaries = [
            0.0,
            2.0 * length,
            8.0 * length,
            24.0 * length / (1.0 - beta),
        ]
    action = _gauss_integral(integrand, boundaries)
    if not np.isfinite(action) or action <= 0.0:
        raise _ConstructionError(
            "nonpositive_action",
            "变量振幅分支没有得到有限正作用量",
            value=action,
            requirement="S_m > 0",
        )
    return action, {
        "alpha": alpha,
        "beta": beta,
        "L_dimensionless": length,
        "b": b_value,
        "candidate_roots": float(len(candidates)),
    }


def estimate_quartic_action(
    a_h: float,
    a_s: float,
    lambda_h: float,
    lambda_s: float,
    lambda_hs: float,
    *,
    temperature: float | None = None,
    source: str = "free",
) -> ActionEstimate:
    """Estimate the action from the five independent quartic coefficients.

    In the four-dimensional thermal convention the returned ``action`` is
    ``S3``.  If ``temperature`` is supplied, ``s3_over_t`` is also returned.
    For dimensionally reduced three-dimensional inputs, ``action`` is already
    the dimensionless exponent ``B`` and ``temperature`` should be omitted.
    """

    parameters = QuarticParameters(a_h, a_s, lambda_h, lambda_s, lambda_hs)
    shape, branch, failures = _validate_quartic(parameters)
    if temperature is not None and (not np.isfinite(temperature) or temperature <= 0.0):
        failures.append(
            FailedCondition(
                "invalid_temperature",
                "temperature 必须是有限正数",
                float(temperature),
                "T > 0",
                "input",
            )
        )
    if failures:
        return ActionEstimate(
            False, None, None, branch, tuple(failures), shape, parameters, source, {}
        )

    potential = _dimensionless_potential(shape)
    try:
        if branch == "conic":
            reduced_action, diagnostics = _conic_action(potential, shape)
        else:
            reduced_action, diagnostics = _moment_action(potential, shape)
    except _ConstructionError as error:
        return ActionEstimate(
            False,
            None,
            None,
            branch,
            (error.failure,),
            shape,
            parameters,
            source,
            {},
        )

    action = sqrt(a_s) / lambda_s * reduced_action
    exponent = action / temperature if temperature is not None else None
    diagnostics = {"reduced_action": reduced_action, **diagnostics}
    return ActionEstimate(
        True,
        action,
        exponent,
        branch,
        (),
        shape,
        parameters,
        source,
        diagnostics,
    )


def estimate_ssm_action(
    m_s: float,
    lambda_s: float,
    lambda_hs: float,
    temperature: float,
    *,
    v_0: float = 246.0,
    m_h: float = 125.1,
    g: float = 0.653,
    g_prime: float = 0.358,
    y_t: float = 0.995,
    require_history: bool = True,
) -> ActionEstimate:
    """Estimate ``S3`` and ``S3/T`` for the paper's Z2 singlet model."""

    model = SSMParameters(
        m_s, lambda_s, lambda_hs, temperature, v_0, m_h, g, g_prime, y_t
    )
    failures: list[FailedCondition] = []
    for name, value in asdict(model).items():
        _finite_positive(name, value, failures)
    if failures:
        return ActionEstimate(
            False, None, None, None, tuple(failures), {}, None, "ssm", {}
        )

    lambda_h = m_h**2 / (2.0 * v_0**2)
    mu_h_sq = m_h**2 / 2.0
    mu_s_sq = lambda_hs * v_0**2 / 2.0 - m_s**2
    c_h = (
        (3.0 * g**2 + g_prime**2) / 16.0
        + lambda_h / 2.0
        + y_t**2 / 4.0
        + lambda_hs / 24.0
    )
    c_s = lambda_hs / 6.0 + lambda_s / 4.0
    if require_history:
        history_checks = [
            (
                mu_s_sq > 0.0,
                "ssm_singlet_mass",
                "零温参数不能产生论文要求的 singlet 相热历史",
                mu_s_sq,
                "mu_s^2 > 0",
            ),
            (
                lambda_hs > 2.0 * sqrt(lambda_h * lambda_s),
                "ssm_barrier",
                "混合耦合不足以在两轴相之间形成势垒",
                lambda_hs,
                "lambda_hs > 2 sqrt(lambda_h lambda_s)",
            ),
            (
                mu_s_sq / c_s > mu_h_sq / c_h,
                "ssm_phase_order",
                "降温时不是 singlet 方向先发生失稳",
                mu_s_sq / c_s - mu_h_sq / c_h,
                "mu_s^2/c_s > mu_h^2/c_h",
            ),
            (
                mu_s_sq / sqrt(lambda_s) < mu_h_sq / sqrt(lambda_h),
                "ssm_zero_temperature_vacuum",
                "零温时 Higgs 轴真空不是更深的真空",
                mu_s_sq / sqrt(lambda_s) - mu_h_sq / sqrt(lambda_h),
                "mu_s^2/sqrt(lambda_s) < mu_h^2/sqrt(lambda_h)",
            ),
        ]
        failures.extend(
            FailedCondition(
                code,
                message,
                float(value),
                requirement,
                "thermal_history",
            )
            for passed, code, message, value, requirement in history_checks
            if not passed
        )
    if failures:
        return ActionEstimate(
            False, None, None, None, tuple(failures), {}, None, "ssm", {}
        )

    a_h = mu_h_sq - c_h * temperature**2
    a_s = mu_s_sq - c_s * temperature**2
    result = estimate_quartic_action(
        a_h,
        a_s,
        lambda_h,
        lambda_s,
        lambda_hs,
        temperature=temperature,
        source="ssm",
    )
    diagnostics = {
        "mu_h_sq": mu_h_sq,
        "mu_s_sq": mu_s_sq,
        "c_h": c_h,
        "c_s": c_s,
        **result.diagnostics,
    }
    return ActionEstimate(
        result.valid,
        result.action,
        result.s3_over_t,
        result.branch,
        result.failures,
        result.shape,
        result.coefficients,
        "ssm",
        diagnostics,
    )
