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

## Requirements

Python 3.11 or newer with NumPy. No package installation or project setup is
required. Download `2FAST.py` and run it directly.

## Command-line use

Five independent coefficients:

```bash
python 2FAST.py free \
  --a-h 1.1 --a-s 1 \
  --lambda-h 1 --lambda-s 1 --lambda-hs 3
```

Singlet-model coefficients:

```bash
python 2FAST.py ssm \
  --m-s 100 --lambda-s 0.1 \
  --lambda-hs 0.46 --temperature 80
```

Add `--json` for machine-readable output.

For four-dimensional thermal coefficients, `action` is \(S_3\). For
dimensionally reduced three-dimensional coefficients, `action` is already the
dimensionless exponent \(B\).

## Scope

The reliability cuts are empirical conditions from the paper rather than
physical phase boundaries or rigorous error bounds. The program calculates the
bounce-action exponent only. It does not calculate a one-loop prefactor.
