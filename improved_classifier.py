#!/usr/bin/env python3
"""FAQ question classifier with mock/API modes and evaluation support."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


LABELS = ("退款退货", "物流查询", "账号问题", "商品咨询", "投诉建议", "其他")
LABEL_SET = set(LABELS)

SYSTEM_PROMPT = """你是客服 FAQ 分类器。你的任务是把用户问题分到且仅分到一个标准标签。

必须遵守：
1. 只能输出 JSON 对象，格式为 {"category":"标签名"}。
2. category 必须是以下标签之一：退款退货、物流查询、账号问题、商品咨询、投诉建议、其他。
3. 不要输出解释、标点外文本、多个标签或置信度。
"""

USER_PROMPT_TEMPLATE = """请根据分类定义判断用户问题所属客服组。

分类定义：
- 退款退货：用户要求退款、退货、换货，或咨询退款进度。例如：我要退货、钱什么时候退回来、怎么换货。
- 物流查询：用户询问包裹位置、配送状态、快递信息、改派地址、签收异常。例如：快递到哪了、包裹显示签收但没收到、能不能改配送地址。
- 账号问题：用户遇到登录、密码、账号安全、绑定手机号等问题。例如：密码忘了、账号被冻结、异地登录提醒。
- 商品咨询：用户询问商品信息、规格、材质、库存、价格、适用场景。例如：尺码怎么选、是否真皮、充电宝能否带上飞机。
- 投诉建议：用户对服务、商品质量、流程表达不满，要求投诉/举报，或提出产品服务建议。例如：服务太差、质量有问题、建议增加功能。
- 其他：闲聊、确认、感谢、无意义输入，或无法归入以上类别。

冲突规则：
- 退款进度、退货流程、取消退货、部分退货优先归入“退款退货”。
- 快递位置、配送地址、签收异常、快递柜异常优先归入“物流查询”。
- 明确包含投诉、举报、建议，或对质量/流程/服务表达强烈不满时，归入“投诉建议”。
- 同时涉及多个类别时，按用户主要诉求判断；若主要诉求不明确，选择最需要客服组处理的类别。

用户问题：{question}
"""


@dataclass(frozen=True)
class Prediction:
    category: str
    raw: str = ""
    error: str = ""


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def normalize_category(raw: str | None) -> str | None:
    """Extract and canonicalize a category from model text."""
    if not raw:
        return None

    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            text = str(data.get("category", "")).strip()
    except json.JSONDecodeError:
        pass

    aliases = {
        "退换货": "退款退货",
        "退款": "退款退货",
        "退货": "退款退货",
        "售后": "退款退货",
        "物流": "物流查询",
        "快递": "物流查询",
        "配送": "物流查询",
        "账号": "账号问题",
        "账户问题": "账号问题",
        "登录问题": "账号问题",
        "商品": "商品咨询",
        "产品咨询": "商品咨询",
        "商品问题": "商品咨询",
        "投诉": "投诉建议",
        "建议": "投诉建议",
        "投诉与建议": "投诉建议",
        "其它": "其他",
        "无法分类": "其他",
    }

    if text in LABEL_SET:
        return text
    if text in aliases:
        return aliases[text]

    for label in LABELS:
        if label in text:
            return label
    for alias, label in aliases.items():
        if alias in text:
            return label
    return None


def baseline_mock_classify(question: str) -> Prediction:
    """Simulate the current weak prompt with category-order keyword matching."""
    q = question.strip()
    if contains_any(q, ("退款", "退货", "换货", "退掉", "只退", "取消退货")):
        return Prediction("退款退货", "baseline.keyword.refund")
    if contains_any(q, ("快递", "物流", "包裹", "配送", "地址", "签收", "快递柜")):
        return Prediction("物流查询", "baseline.keyword.logistics")
    if contains_any(q, ("账号", "账户", "密码", "登录", "手机号", "绑定", "冻结", "短信")):
        return Prediction("账号问题", "baseline.keyword.account")
    if contains_any(q, ("商品", "尺码", "码", "耳机", "降噪", "鞋", "手机壳", "硅胶", "塑料", "真皮", "充电宝", "飞机")):
        return Prediction("商品咨询", "baseline.keyword.product")
    if contains_any(q, ("投诉", "建议", "举报", "服务", "质量", "太差", "太麻烦", "破质量", "坏了")):
        return Prediction("投诉建议", "baseline.keyword.complaint")
    return Prediction("其他", "baseline.keyword.other")


def improved_mock_classify(question: str) -> Prediction:
    """Deterministic mock of the redesigned prompt and conflict rules."""
    q = question.strip()
    if not q or re.fullmatch(r"[\s？?!.。！,，、~～…]+", q):
        return Prediction("其他", "mock.rule.empty_or_symbol")

    complaint = contains_any(
        q,
        (
            "投诉",
            "举报",
            "建议",
            "服务太差",
            "态度太差",
            "质量有问题",
            "破质量",
            "太麻烦",
            "坏了",
        ),
    )
    refund = contains_any(q, ("退款", "退货", "换货", "退掉", "只退", "取消退货", "退的"))
    logistics = contains_any(q, ("快递", "物流", "包裹", "配送", "地址", "签收", "派送", "快递柜", "寄回去", "寄错"))
    account = contains_any(q, ("账号", "账户", "密码", "登录", "手机号", "绑定", "冻结", "异地登录", "短信"))
    product = contains_any(
        q,
        (
            "商品",
            "尺码",
            "码",
            "耳机",
            "降噪",
            "鞋",
            "手机壳",
            "硅胶",
            "塑料",
            "真皮",
            "充电宝",
            "飞机",
            "材质",
            "库存",
            "价格",
            "补货",
        ),
    )

    if complaint:
        return Prediction("投诉建议", "mock.rule.complaint")
    if refund:
        return Prediction("退款退货", "mock.rule.refund")
    if logistics:
        return Prediction("物流查询", "mock.rule.logistics")
    if account:
        return Prediction("账号问题", "mock.rule.account")
    if product:
        return Prediction("商品咨询", "mock.rule.product")
    return Prediction("其他", "mock.rule.other")


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
    ]


def api_classify(question: str, model: str, timeout: float, retries: int) -> Prediction:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return Prediction("其他", "", "OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI
    except ImportError:
        return Prediction("其他", "", "openai package is not installed")

    client = OpenAI(api_key=api_key, timeout=timeout)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(question),
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            category = normalize_category(raw)
            if category:
                return Prediction(category, raw)
            return Prediction("其他", raw, "model returned an invalid category")
        except Exception as exc:  # noqa: BLE001 - keep batch jobs alive and report per-item failures.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (2**attempt))
    return Prediction("其他", "", last_error)


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("input JSON must be a list")
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} must be an object")
        if "id" not in item or "question" not in item:
            raise ValueError(f"item {index} must contain id and question")
        if not isinstance(item["question"], str):
            raise ValueError(f"item {index} question must be a string")
    return data


def score_rows(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("label")]
    total = len(labeled_rows)
    correct = sum(1 for row in labeled_rows if row.get(prediction_key) == row.get("label"))
    per_label: dict[str, dict[str, int | float]] = {}
    for label in LABELS:
        items = [row for row in labeled_rows if row.get("label") == label]
        label_total = len(items)
        label_correct = sum(1 for row in items if row.get(prediction_key) == label)
        per_label[label] = {
            "correct": label_correct,
            "total": label_total,
            "accuracy": round(label_correct / label_total, 4) if label_total else 0.0,
        }
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_label": per_label,
    }


def evaluate(
    samples: list[dict[str, Any]],
    improved_classifier: Callable[[str], Prediction],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()

    for item in samples:
        question = item["question"]
        label = item.get("label")
        baseline = baseline_mock_classify(question)
        improved = improved_classifier(question)
        row = {
            "id": item["id"],
            "question": question,
            "label": label,
            "baseline_pred": baseline.category,
            "baseline_raw": baseline.raw,
            "improved_pred": improved.category,
            "improved_raw": improved.raw,
            "error": improved.error,
            "baseline_correct": baseline.category == label if label else None,
            "improved_correct": improved.category == label if label else None,
        }
        rows.append(row)
        if label:
            confusion[(label, improved.category)] += 1

    return {
        "labels": list(LABELS),
        "total_samples": len(samples),
        "baseline": score_rows(rows, "baseline_pred"),
        "improved": score_rows(rows, "improved_pred"),
        "confusion_matrix": [
            {"label": label, "prediction": pred, "count": count}
            for (label, pred), count in sorted(confusion.items())
        ],
        "rows": rows,
    }


def classify_batch(
    samples: list[dict[str, Any]],
    classifier: Callable[[str], Prediction],
) -> list[dict[str, Any]]:
    results = []
    for item in samples:
        prediction = classifier(item["question"])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "predicted_category": prediction.category,
                "raw": prediction.raw,
                "error": prediction.error,
            }
        )
    return results


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def print_eval_summary(result: dict[str, Any]) -> None:
    baseline = result["baseline"]
    improved = result["improved"]
    print("FAQ 分类评估结果")
    print(f"样本数: {result['total_samples']}")
    print(f"Baseline: {baseline['correct']}/{baseline['total']} = {baseline['accuracy']:.2%}")
    print(f"Improved: {improved['correct']}/{improved['total']} = {improved['accuracy']:.2%}")
    print("\n错分明细（改进后）:")
    misses = [row for row in result["rows"] if row.get("improved_correct") is False]
    if not misses:
        print("- 无")
    for row in misses:
        print(f"- #{row['id']} label={row['label']} pred={row['improved_pred']} question={row['question']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="客服 FAQ 自动分类脚本")
    parser.add_argument("--input", required=True, type=Path, help="输入 JSON 文件")
    parser.add_argument("--output", required=True, type=Path, help="输出 JSON 文件")
    parser.add_argument("--mode", choices=("mock", "api"), default="mock", help="分类模式")
    parser.add_argument("--evaluate", action="store_true", help="计算准确率对比")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="API 模型名")
    parser.add_argument("--timeout", type=float, default=20.0, help="API 超时时间（秒）")
    parser.add_argument("--retries", type=int, default=2, help="API 失败重试次数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    samples = load_samples(args.input)

    if args.mode == "api":
        classifier = lambda question: api_classify(question, args.model, args.timeout, args.retries)
    else:
        classifier = improved_mock_classify

    if args.evaluate:
        result = evaluate(samples, classifier)
        write_json(args.output, result)
        print_eval_summary(result)
    else:
        results = classify_batch(samples, classifier)
        write_json(args.output, results)
        print(f"分类完成，共处理 {len(results)} 条问题，结果写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
