"""赛制评分层：按郑商所杯**第九届官方细则**计算得分。

官方原文
========

郑州商品交易所、中国期货业协会《关于举办第九届"郑商所杯"大学生金融衍生品专业能力大赛的通知》
**郑商函〔2026〕670号**（2026 年 7 月 14 日）规定：

    4.1 模拟交易测试
    （1）计分规则：收益率得分 × 80% + 回撤得分 × 10% + 活跃度得分 × 10%
    （2）账户初始虚拟保证金为 50 万元
    （3）交易品种：郑商所已上市所有品种，包括期货合约和期权合约
    （4）交易制度：与郑商所实盘交易一致。期货合约将在其交割月前一个交易日停止交易
    （5）保证金按照郑商所规定标准收取；手续费按照郑商所规定的 1.5 倍收取
    （6）大赛最后一天，参赛者可以继续保留持仓，当日权益按照收盘价计算
    （7）活跃交易日要求：每个账户活跃交易日不得少于 5 天。不满足的账户无法参与最终评奖

    4.2 理论知识测试
    （1）通过期货从业人员资格考试（期货基础知识 + 期货法律法规）
        或通过 FDA Ⅰ级课程线上专项测试，获得理论知识测试得分 **5 分**，5 分封顶
    （2）须在 2026 年 11 月 20 日前通过

    4.3 综合评定：最终成绩 = 模拟交易测试得分 + 理论附加分

**三条被本仓库早期版本搞错的事实，已经改正：**

1. 第三项是 **活跃度得分 10%**，**不是波动率** —— 所以"为了活跃度而多交易"
   与"1.5 倍手续费"之间的张力是**真实存在**的。
2. 理论附加分是 **5 分**，不是 10 分。
3. **活跃交易日 ≥ 5 天是硬门槛**：不满足直接失去评奖资格。
   这意味着"完全不交易的保守策略"会被判出局 —— 上限不是加分，而是资格。

官方通知没有公开**每一部分如何归一化**（也没有公开"收益率得分"的具体算法）。
所以归一化参数继续保持可调并做敏感性分析。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

THEORY_BONUS = 5.0
"""通过期货从业资格考试（两门）或 FDA Ⅰ级线上专项测试可获得的固定附加分，5 分封顶。"""

MIN_ACTIVE_DAYS = 5
"""活跃交易日的**资格门槛**：少于 5 天无法参与最终评奖。"""

FEE_MULTIPLIER = 1.5
"""官方规定：手续费按郑商所规定的 1.5 倍收取。"""


@dataclass
class ScoreAssumptions:
    """赛制评分的假设参数。**权重来自官方细则**，归一化参数是估计值。"""

    return_weight: float = 0.80
    drawdown_weight: float = 0.10
    activity_weight: float = 0.10
    return_reference: float = 0.20
    """收益率得分拿满所需的总收益率。"""
    drawdown_tolerance: float = 0.15
    """最大回撤到此水平时回撤项归零。"""
    activity_target: float = 0.30
    """活跃度得分拿满所需的有成交交易日占比。"""
    name: str = "第九届官方权重（80/10/10）"

    def weight_sum(self) -> float:
        """三项权重之和，用于自检（应为 1.0）。"""
        return self.return_weight + self.drawdown_weight + self.activity_weight

    def describe(self) -> str:
        """中文描述，写进报告。"""
        return (
            f"{self.name}：收益率满分线 {self.return_reference:.0%}、"
            f"回撤归零线 {self.drawdown_tolerance:.0%}、"
            f"活跃度目标 {self.activity_target:.0%}；"
            f"权重 {self.return_weight:.0%}/{self.drawdown_weight:.0%}/{self.activity_weight:.0%}"
        )


DEFAULT_ASSUMPTIONS: tuple[ScoreAssumptions, ...] = (
    ScoreAssumptions(name="中性假设"),
    ScoreAssumptions(
        name="收益率线低（容易拿分）",
        return_reference=0.10, drawdown_tolerance=0.20, activity_target=0.20,
    ),
    ScoreAssumptions(
        name="收益率线高（难拿分）",
        return_reference=0.40, drawdown_tolerance=0.10, activity_target=0.50,
    ),
    ScoreAssumptions(
        name="重活跃度（门槛提高）",
        return_weight=0.70, drawdown_weight=0.10, activity_weight=0.20, activity_target=0.50,
    ),
    ScoreAssumptions(
        name="轻活跃度（只求过 5 天门槛）",
        return_weight=0.89, drawdown_weight=0.10, activity_weight=0.01, activity_target=0.30,
    ),
)
"""一组覆盖范围的假设，用来检验结论的稳健性。"""


def score_components(metrics: dict[str, Any], assumptions: ScoreAssumptions = ScoreAssumptions()) -> dict[str, Any]:
    """把回测指标算成三个分项与加权总分（0-1 量纲，乘 100 就是交易能力测试得分）。"""
    total_return = float(metrics.get("总收益率", 0.0) or 0.0)
    max_drawdown = float(metrics.get("最大回撤", 0.0) or 0.0)
    active_ratio = float(metrics.get("活跃交易日占比", 0.0) or 0.0)

    return_score = float(np.clip(total_return / assumptions.return_reference, -1.0, 1.0))
    drawdown_score = float(np.clip(1.0 - max_drawdown / assumptions.drawdown_tolerance, 0.0, 1.0))
    activity_score = float(np.clip(active_ratio / assumptions.activity_target, 0.0, 1.0))
    composite = (
        assumptions.return_weight * return_score
        + assumptions.drawdown_weight * drawdown_score
        + assumptions.activity_weight * activity_score
    )
    return {
        "假设": assumptions.name,
        "收益率分项": round(return_score, 4),
        "回撤分项": round(drawdown_score, 4),
        "活跃度分项": round(activity_score, 4),
        "模拟总分": round(composite, 4),
        "总分×100": round(composite * 100, 2),
        "含附加分": round(composite * 100 + THEORY_BONUS, 2),
        "总收益率": round(total_return, 4),
        "最大回撤": round(max_drawdown, 4),
        "活跃交易日占比": round(active_ratio, 4),
        "达到活跃门槛": bool(metrics.get("有交易天数", 0) >= MIN_ACTIVE_DAYS),
        "_假设参数": asdict(assumptions),
    }


def score_sensitivity(
    metrics: dict[str, Any], assumptions: Iterable[ScoreAssumptions] = DEFAULT_ASSUMPTIONS
) -> pd.DataFrame:
    """对同一份回测结果跑多组假设，看总分区间有多宽。"""
    return pd.DataFrame([score_components(metrics, item) for item in assumptions])


def rank_robustness(
    results: dict[str, dict[str, Any]],
    assumptions: Iterable[ScoreAssumptions] = DEFAULT_ASSUMPTIONS,
) -> pd.DataFrame:
    """比较多个候选策略：给出每组假设下的排名与平均排名，找出稳健者。"""
    assumption_list = list(assumptions)
    rows: list[dict[str, Any]] = []
    for name, metrics in results.items():
        row: dict[str, Any] = {"策略": name}
        scores = []
        for assumption in assumption_list:
            value = score_components(metrics, assumption)["模拟总分"]
            row[assumption.name] = round(value, 4)
            scores.append(value)
        row["平均总分"] = round(float(np.mean(scores)), 4)
        row["最差总分"] = round(float(np.min(scores)), 4)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for assumption in assumption_list:
        frame[f"{assumption.name}_排名"] = frame[assumption.name].rank(ascending=False, method="min").astype(int)
    rank_columns = [f"{assumption.name}_排名" for assumption in assumption_list]
    frame["平均排名"] = frame[rank_columns].mean(axis=1).round(2)
    return frame.sort_values(["平均排名", "平均总分"], ascending=[True, False]).reset_index(drop=True)


def optimal_frequency_note(table: pd.DataFrame, score_column: str = "模拟总分") -> str:
    """给出一句结论：最优交易频率在哪里。

    **这是赛制真正的张力点**：活跃度占 10% 要求你交易，
    1.5 倍手续费要求你少交易，而 80% 权重压在这两件事的结果上。
    """
    if table.empty:
        return "没有可比较的参数组合。"
    best = table.loc[table[score_column].idxmax()]
    cheapest = table.loc[table["手续费合计"].idxmin()]
    most = table.loc[table["交易笔数"].idxmax()]
    return (
        f"模拟总分最高的是「{best['参数']}」（{best[score_column]:.4f}，"
        f"{int(best['交易笔数'])} 笔交易，手续费 {best['手续费合计']:,.0f} 元）；"
        f"最省手续费的是「{cheapest['参数']}」（{cheapest['手续费合计']:,.0f} 元）；"
        f"交易最多的是「{most['参数']}」（{int(most['交易笔数'])} 笔，"
        f"总分 {most[score_column]:.4f}）。"
        "把三者放在一起看，就能判断「为了活跃度而多交易」到底划不划算。"
    )


def entry_requirement_note() -> str:
    """把硬门槛与附加分单独说清楚 —— 这两条与策略无关，却是确定的得失。"""
    return (
        f"1. **活跃交易日 ≥ {MIN_ACTIVE_DAYS} 天是资格门槛**，不是加分项。"
        "达不到直接无法参与评奖。所以「完全不交易」不是保守，是弃权。\n"
        f"2. **理论附加分 {THEORY_BONUS:.0f} 分是确定的**：在 2026 年 11 月 20 日前"
        "通过期货从业人员资格考试（两门）或 FDA Ⅰ级课程线上专项测试即可。"
        "而我们的回测显示，全部参数调整带来的分差通常不到 5 分 —— "
        "**考一次试的收益大于所有策略优化的总和，而且没有亏损风险。**"
    )
