"""Command-line interface for the two-field quartic action estimator."""

from __future__ import annotations

import argparse
import json

from twofast.estimator import estimate_quartic_action, estimate_ssm_action


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="2fast",
        description="论文中的双标量场四次势非迭代作用量估算器",
    )
    subparsers = parser.add_subparsers(dest="input_type", required=True)

    free = subparsers.add_parser("free", help="输入五个独立势参数")
    free.add_argument("--a-h", type=float, required=True)
    free.add_argument("--a-s", type=float, required=True)
    free.add_argument("--lambda-h", type=float, required=True)
    free.add_argument("--lambda-s", type=float, required=True)
    free.add_argument("--lambda-hs", type=float, required=True)
    free.add_argument("--temperature", type=float)
    free.add_argument("--json", action="store_true")

    ssm = subparsers.add_parser("ssm", help="输入 Z2 singlet model 参数")
    ssm.add_argument("--m-s", type=float, required=True)
    ssm.add_argument("--lambda-s", type=float, required=True)
    ssm.add_argument("--lambda-hs", type=float, required=True)
    ssm.add_argument("--temperature", type=float, required=True)
    ssm.add_argument("--ignore-history", action="store_true")
    ssm.add_argument("--json", action="store_true")
    return parser


def _print_human(result) -> None:
    print(f"满足全部判断条件: {'是' if result.valid else '否'}")
    if result.shape:
        print(
            "形状参数: "
            + ", ".join(f"{name}={value:.8g}" for name, value in result.shape.items())
        )
    if result.branch:
        print(f"估算分支: {result.branch}")
    if result.valid:
        print(f"S = {result.action:.12g}")
        if result.s3_over_t is not None:
            print(f"S3/T = {result.s3_over_t:.12g}")
        return
    print("未通过的条件:")
    for failure in result.failures:
        detail = (
            f"，当前值={failure.value:.8g}" if failure.value is not None else ""
        )
        required = f"，要求 {failure.requirement}" if failure.requirement else ""
        print(
            f"- [{failure.stage}/{failure.code}] "
            f"{failure.message}{detail}{required}"
        )


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.input_type == "free":
        result = estimate_quartic_action(
            arguments.a_h,
            arguments.a_s,
            arguments.lambda_h,
            arguments.lambda_s,
            arguments.lambda_hs,
            temperature=arguments.temperature,
        )
    else:
        result = estimate_ssm_action(
            arguments.m_s,
            arguments.lambda_s,
            arguments.lambda_hs,
            arguments.temperature,
            require_history=not arguments.ignore_history,
        )
    if arguments.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
