from __future__ import annotations

import pytest

from twofast import estimate_quartic_action, estimate_ssm_action


def test_free_parameters_return_moment_action_in_validated_domain():
    result = estimate_quartic_action(
        a_h=1.1,
        a_s=1.0,
        lambda_h=1.0,
        lambda_s=1.0,
        lambda_hs=3.0,
    )

    assert result.valid
    assert result.branch == "moment"
    assert result.action == pytest.approx(207.858807827, rel=2.0e-8)
    assert result.s3_over_t is None
    assert result.failures == ()


def test_free_parameters_return_conic_action_in_validated_domain():
    result = estimate_quartic_action(
        a_h=1.3,
        a_s=1.0,
        lambda_h=1.0,
        lambda_s=1.0,
        lambda_hs=5.0,
    )

    assert result.valid
    assert result.branch == "conic"
    assert result.action is not None and result.action > 0.0
    assert result.diagnostics["c"] >= 0.0


def test_action_obeys_the_paper_scaling_factorization():
    reference = estimate_quartic_action(1.1, 1.0, 1.0, 1.0, 3.0)
    rescaled = estimate_quartic_action(9.9, 9.0, 1.0, 1.0, 3.0)

    assert reference.valid and rescaled.valid
    assert rescaled.action == pytest.approx(3.0 * reference.action, rel=2.0e-12)


def test_invalid_state_reports_each_failed_physics_condition_without_action():
    result = estimate_quartic_action(0.8, 1.0, 1.0, 1.0, 1.0)

    assert not result.valid
    assert result.action is None
    codes = {failure.code for failure in result.failures}
    assert "mixed_barrier" in codes
    assert "vacuum_ordering" in codes


def test_outside_empirical_domain_reports_the_specific_domain_cut():
    # gamma=3, eta=0.5, d=0.1 and epsilon=0.2 is physically metastable,
    # but it fails the paper's gradient-cost reliability cut.
    result = estimate_quartic_action(3.3, 1.0, 9.0, 1.0, 9.0)

    assert not result.valid
    assert result.action is None
    assert [failure.code for failure in result.failures] == [
        "validated_gradient_domain"
    ]


def test_ssm_mode_returns_s3_and_thermal_exponent():
    result = estimate_ssm_action(100.0, 0.1, 0.46, 80.0)

    assert result.valid
    assert result.action is not None
    assert result.s3_over_t == pytest.approx(result.action / 80.0)
    assert result.coefficients is not None
    assert result.source == "ssm"


def test_ssm_mode_reports_failed_thermal_history_condition():
    result = estimate_ssm_action(400.0, 0.1, 0.2, 80.0)

    assert not result.valid
    assert result.action is None
    assert any(failure.code.startswith("ssm_") for failure in result.failures)
