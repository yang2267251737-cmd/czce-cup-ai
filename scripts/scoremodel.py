"""赛制评分层：把回测指标映射成"总分 = 收益率 × 80% + 回撤 × 10% + 活跃度 × 10%"。

**重要：这是一套假设模型，不是官方公式。**
公开信息只给出了三部分的**权重**（80/10/10），没有给出每一部分如何归一化。
所以本模块把三个归一化参数（收益率满分线、回撤容忍度、活跃度目标）全部显式参数化，
并提供 :func:`score_sensitivity` 做敏感性分析。

这样做的意义：结论不依赖某一个猜测的公式。如果某个策略在**所有合理假设下**都排在前列，
它才是真正稳健的选择；只在某一组假设下第一的策略，不值得信。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class ScoreAssumptions:
    """赛制评分的假设参数，全部可调，方便做敏感性分析。"""

    return_weight: float = 0.80
    drawdown_weight: float = 0.10
    activity_weight: float = 0.10
    return_reference: float = 0.30
    """收益率拿满分的水平：0.30 表示比赛期内 +30% 记 100 分。"""
    drawdown_tolerance: float = 0.15
    """回撤得 0 分的水平：0.15 表示最大回撤 15% 时回撤项归零。"""
    activity_target: float = 0.35
    """活跃度拿满分所需的有交易交易日占比。"""
    name: str = "基准假设"

    def weight_sum(self) -> float:
        """三项权重之和，用于自检。"""
        return self.return_weight + self.drawdown_weight + self.activity_weight

    def describe(self) -> str:
        """中文描述，写进报告。"""
        return (
            f"{self.name}：收益满分线 {self.return_reference:.0%}、"
            f"回撤容忍度 {self.drawdown_tolerance:.0%}、"
            f"活跃度目标 {self.activity_target:.0%}，权重 "
            f"{self.return_weight:.0%}/{self.drawdown_weight:.0%}/{self.activity_weight:.0%}"
        )


DEFAULT_ASSUMPTIONS: tuple[ScoreAssumptions, ...] = (
    ScoreAssumptions(name="中性假设"),
    ScoreAssumptions(
        name="乐观假设（收益率线低、容忍高回撤）",
        return_reference=0.15, drawdown_tolerance=0.25, activity_target=0.25,
    ),
    ScoreAssumptions(
        name="悲观假设（收益率线高、回撤敏感）",
        return_reference=0.50, drawdown_tolerance=0.08, activity_target=0.50,
    ),
    ScoreAssumptions(
        name="重活跃度假设（活跃度权重翻倍）",
        return_weight=0.70, drawdown_weight=0.10, activity_weight=0.20, activity_target=0.60,
    ),
    ScoreAssumptions(
        name="轻活跃度假设（活跃度几乎不计）",
        return_weight=0.89, drawdown_weight=0.10, activity_weight=0.01, activity_target=0.35,
    ),
)
"""一组覆盖范围的假设，用来检验结论的稳健性。"""


def score_components(metrics: dict[str, Any], assumptions: ScoreAssumptions = ScoreAssumptions()) -> dict[str, Any]:
    """把回测指标算成三个分项与加权总分（0-1 量纲，可乘 100 当"分"看）。"""
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
        "总收益率": round(total_return, 4),
        "最大回撤": round(max_drawdown, 4),
        "活跃交易日占比": round(active_ratio, 4),
        "手续费合计": round(float(metrics.get("手续费合计", 0.0) or 0.0), 2),
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
    """比较多个候选策略：给出每组假设下的排名与平均排名，找出稳健者。

    ``results`` 形如 ``{策略名: 回测指标字典}``。平均排名最小的策略最稳健。
    """
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


def fee_activity_tradeoff(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """把"交易频率 → 收益 / 手续费 / 活跃度"的权衡整理成表。

    ``rows`` 每一项包含 ``参数``、``交易笔数``、``手续费合计``、``活跃交易日占比`` 与回测指标。
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["每笔平均手续费"] = (frame["手续费合计"] / frame["交易笔数"].replace(0, np.nan)).round(2)
    return frame


def optimal_frequency_note(table: pd.DataFrame, score_column: str = "模拟总分") -> str:
    """给出一句结论：最优交易频率在哪里，以及活跃度是否值得为它多交易。"""
    if table.empty:
        return "没有可比较的参数组合。"
    best = table.loc[table[score_column].idxmax()]
    low_fee = table.loc[table["手续费合计"].idxmin()]
    high_activity = table.loc[table["活跃交易日占比"].idxmax()]
    return (
        f"模拟总分最高的是「{best['参数']}」（总分 {best[score_column]:.4f}，"
        f"{int(best['交易笔数'])} 笔交易，手续费 {best['手续费合计']:,.0f} 元）；"
        f"手续费最低的是「{low_fee['参数']}」（{low_fee['手续费合计']:,.0f} 元）；"
        f"活跃度最高的是「{high_activity['参数']}」（有交易交易日占比 "
        f"{high_activity['活跃交易日占比']:.1%}，但总分 {high_activity[score_column]:.4f}）。"
        "把三者放在一起看，就能判断「为了活跃度而多交易」到底划不划算。"
    )
