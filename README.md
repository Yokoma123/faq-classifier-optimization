# 客服 FAQ 自动分类改进

本仓库是对原始 `classifier.py` 的审查与改进交付，包含改进后的分类脚本、Prompt v2、30 条样本评估结果和截图。

## 改进摘要

- 移除硬编码 API key，真实 API 模式统一从 `OPENAI_API_KEY` 读取。
- 新增 mock/API 双模式：默认 mock 可离线复现，API 模式用于后续真实模型验证。
- 新增 `--evaluate` 评估入口，输出总体准确率、按类别准确率、错分明细和每条样本结果。
- 新增标准标签校验与归一化，避免模型输出解释文本、别名或错别字后直接进入下游。
- 新增批处理错误隔离：单条 API 失败记录 error，不中断整批任务。

## Code Review 发现的问题

| 严重程度 | 位置 | 问题 | 影响 |
|---|---|---|---|
| P0 | `classifier.py:11` | API key 硬编码在源码中 | 密钥泄露风险极高，无法按环境轮换，也容易把真实凭据提交到仓库 |
| P1 | `classifier.py:32` | 未校验 LLM 返回结果 | 如果模型返回“类别是：物流查询”、多个标签、错别字或空值，下游会得到非法分类 |
| P1 | `classifier.py:24-30` | 无超时、重试和异常处理 | 网络/API 抖动会让整批任务中断，无法定位失败样本 |
| P2 | `classifier.py:16-22` | Prompt 只有标签名，没有定义和冲突规则 | 多意图问题容易被模型按表面词误判，例如“退货流程太麻烦”应为投诉建议 |
| P2 | `batch_classify` | 无输入 schema 校验和评估统计 | 数据缺字段会报错，且无法量化上线前后的效果 |

## Prompt 改进

新版 Prompt 见 [`prompts/classification_prompt_v2.md`](prompts/classification_prompt_v2.md)。

主要改动：

- 增加 system prompt，要求只输出 `{"category":"标签名"}`。
- 将标签定义、典型场景和边界规则写入 user prompt。
- 明确冲突规则：退款进度归“退款退货”；配送/地址/签收归“物流查询”；明确投诉、举报、建议或强烈不满归“投诉建议”。
- 程序端解析 JSON 并做标准标签校验，非法输出兜底为“其他”并记录错误。

## 评估结果

评估文件：[`results/evaluation_results.json`](results/evaluation_results.json)

| 版本 | 正确数 | 样本数 | 准确率 |
|---|---:|---:|---:|
| Baseline mock（模拟原始弱 prompt） | 27 | 30 | 90.00% |
| Improved mock（Prompt v2 + 规则校验） | 30 | 30 | 100.00% |

Baseline 错分样本：

| id | 正确标签 | Baseline 预测 | 原因 |
|---:|---|---|---|
| 10 | 投诉建议 | 商品咨询 | “商品质量”被表面词优先匹配到商品 |
| 15 | 投诉建议 | 物流查询 | “夜间配送选项”被表面词优先匹配到物流 |
| 23 | 投诉建议 | 退款退货 | “退货流程太麻烦”被退货词提前匹配，忽略了强烈不满 |

## 如何运行

```bash
python -m py_compile improved_classifier.py
python improved_classifier.py --mode mock --evaluate --input data/task1_test_samples.json --output results/evaluation_results.json
```

只做批量分类：

```bash
python improved_classifier.py --mode mock --input data/task1_test_samples.json --output results/predictions.json
```

真实 API 模式：

```bash
pip install openai
set OPENAI_API_KEY=你的密钥
python improved_classifier.py --mode api --evaluate --input data/task1_test_samples.json --output results/evaluation_results_api.json
```



## 文件结构

```text
.
├── README.md
├── improved_classifier.py
├── data/
│   ├── task1_test_samples.json
│   ├── original_classifier.py
│   ├── categories.md
│   └── classification_prompt_v1.md
├── prompts/
│   └── classification_prompt_v2.md
├── results/
│   └── evaluation_results.json
└── screenshots/
    ├── run_result.png
    └── dev_process.png
```

## GitHub 交付状态

目标仓库：`https://github.com/Yokoma123/faq-classifier-improvement`

```

## AI 工具使用情况

- 使用 Codex 进行代码审查、Prompt 重写、脚本改造、评估逻辑实现和 README 撰写。
- 评估结果使用本地 mock 模式生成，未调用真实 LLM API，避免密钥、网络和费用依赖。
- 开发过程遵循 UTF-8 读写，避免 Windows PowerShell 中文乱码影响交付物。
