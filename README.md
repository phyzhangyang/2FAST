# 2FAST

**Two-Field Action Semianalytic Tool**

2FAST implements the non-iterative semianalytic estimator for the thermal
tunneling action in a two-field quartic potential

\[
V(h,s)=-\frac{a_h}{2}h^2-\frac{a_s}{2}s^2
+\frac{\lambda_h}{4}h^4+\frac{\lambda_s}{4}s^4
+\frac{\lambda_{hs}}{4}h^2s^2.
\]

It supports two input modes.

- `free` takes the five independent potential coefficients
  \((a_h,a_s,\lambda_h,\lambda_s,\lambda_{hs})\).
- `ssm` takes \((m_s,\lambda_s,\lambda_{hs},T)\) for the leading thermal-mass
  potential of the Z2-symmetric real-singlet extension of the Standard Model.

The program first checks the physical metastability conditions, the empirical
reliability domain established in the paper, and the construction conditions
of the selected conic or moment branch. It returns an action only when all
applicable conditions pass. Otherwise it reports every failed condition.

## Installation

```bash
python -m pip install -e .
```

## Command-line use

Five independent coefficients:

```bash
2fast free \
  --a-h 1.1 --a-s 1 \
  --lambda-h 1 --lambda-s 1 --lambda-hs 3
```

Singlet-model coefficients:

```bash
2fast ssm \
  --m-s 100 --lambda-s 0.1 \
  --lambda-hs 0.46 --temperature 80
```

The source checkout can also be run without installation:

```bash
python 2FAST.py free \
  --a-h 1.1 --a-s 1 \
  --lambda-h 1 --lambda-s 1 --lambda-hs 3
```

Add `--json` for machine-readable output.

## Python API

```python
from twofast import estimate_quartic_action, estimate_ssm_action

free = estimate_quartic_action(1.1, 1.0, 1.0, 1.0, 3.0)
ssm = estimate_ssm_action(100.0, 0.1, 0.46, 80.0)

if ssm.valid:
    print(ssm.action)       # S3
    print(ssm.s3_over_t)   # S3/T
else:
    for failure in ssm.failures:
        print(failure.stage, failure.code, failure.message)
```

For four-dimensional thermal coefficients, `action` is \(S_3\). For
dimensionally reduced three-dimensional coefficients, `action` is already the
dimensionless exponent \(B\).

## Scope

The reliability cuts are empirical conditions from the paper rather than
physical phase boundaries or rigorous error bounds. The program calculates the
bounce-action exponent only. It does not calculate a one-loop prefactor.
