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

The program first checks the physical metastability conditions, the refined
empirical reliability domain, and the construction conditions of the selected
conic or moment branch. It returns an action only when all applicable
conditions pass. Otherwise it reports every failed condition.

The dimensionless shape variables used by the estimator are

\[
\gamma=\sqrt{\frac{\lambda_h}{\lambda_s}},\qquad
\eta=\frac{\lambda_{hs}}{2\sqrt{\lambda_h\lambda_s}}-1,\qquad
d=\frac{a_h}{\gamma a_s}-1,\qquad
\epsilon=\frac{d}{\eta}.
\]

The current prescription uses the conic construction for
\(\eta\geq0.4\) when \(\epsilon<0.4\). The variable-amplitude branch uses the
soft-limit interpolation weight \(\epsilon^{3/2}\). Its refined domain
requires \(\eta<0.8\) and excludes the corner \(\eta>0.7\),
\(\gamma>1.5\), and \(\epsilon>0.8\). The complete expressions are given in
the associated work.

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

The reliability cuts and numerical coefficients of the estimator are
empirical rather than physical phase boundaries or rigorous error bounds. They
were developed on a 10,000-point sample and checked on a fresh 5,000-point
sample after the unified conic factor was fixed. The mean absolute relative
error against PhaseTracer was 0.39 percent, the 95th percentile was 1.36
percent, and 4,998 of the 5,000 actions were within 5 percent. For the 3,251
conic points, the corresponding values were 0.25 percent, 0.63 percent, and
100 percent. PhaseTracer tightening and independent CosmoTransitions checks
were used to test the large-error tail in the preceding validation.

The program calculates the bounce-action exponent only. It does not calculate
a one-loop prefactor.
