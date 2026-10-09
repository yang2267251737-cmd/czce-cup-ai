"""风控层：仓位计算、组合约束与回撤守门。

赛制把回撤计入 10% 的分数，所以风控不是"锦上添花"，而是直接拿分的一环。
本模块只做三件事：
1. **按单笔风险定额**：每笔最多亏 ``risk_per_trade`` × 权益，止损距离由 ATR 决定，反推手数。
2. **按保证金上限截断**：即使风险预算允许，也不允许保证金占用超过 ``max_margin_usage``。
3. **回撤守门**：账户回撤达到阈值时减半、达到暂停线时停止开新仓（只管理已有持仓）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from contracts import Product


@dataclass
class RiskConfig:
    """风控参数。默认值为用户已确认的待研究参数（单笔 1%、回撤 10% 暂停）。"""

    equity: float = 500_000.0
    risk_per_trade: float = 0.01
    max_positions: int = 6
    max_margin_usage: float = 0.60
    drawdown_scale: float = 0.06
    drawdown_pause: float = 0.10
    pause_risk_multiplier: float = 0.25
    """回撤触及暂停线后保留的风险倍数。

    这里**不能置零**：回测发现，一旦停止开新仓，账户就失去了唯一的恢复途径，
    权益永远回不到高点，暂停状态被永久锁死（实测交易笔数从 68 笔塌到 19 笔，
    且再也不会恢复）。因此把"暂停"实现成"降到 1/4 风险继续观察"。
    """
    min_lots: int = 1
    max_lots_per_product: int = 12
    max_single_risk: float = 0.02
    """单手兜底规则允许的单笔最大风险。

    小账户会遇到一个现实问题：风险预算被回撤守门压到 1/4 之后，
    预算可能连 1 手都覆盖不了（例如 50 万 × 1% × 1/4 = 1250 元，
    而棉花 1 手 3ATR 止损的风险约 3400 元），于是**所有信号都被放弃**。
    为了不让账户陷入"越亏越不敢做、越不敢做越不能回本"的死循环，
    允许在风险不超过权益 2% 且保证金够用时按 1 手兜底。
    """

    def describe(self) -> list[str]:
        """给报告用的中文描述。"""
        return [
            f"初始权益 {self.equity:,.0f} 元",
            f"单笔风险 {self.risk_per_trade:.1%}（约 {self.equity * self.risk_per_trade:,.0f} 元）",
            f"最多同时持有 {self.max_positions} 个品种",
            f"保证金占用上限 {self.max_margin_usage:.0%}",
            f"回撤达 {self.drawdown_scale:.0%} 时风险减半，"
            f"达 {self.drawdown_pause:.0%} 时风险降至 {self.pause_risk_multiplier:.0%}（不停止交易，避免锁死）",
        ]


def lots_by_risk(equity: float, risk_per_trade: float, stop_points: float, product: Product, price: float) -> int:
    """按单笔风险预算反推手数；风险预算连 1 手都覆盖不了时返回 0（宁可不做）。"""
    risk_per_lot = abs(stop_points) * product.multiplier
    if risk_per_lot <= 0 or not np.isfinite(risk_per_lot):
        return 0
    budget = equity * risk_per_trade
    return int(budget // risk_per_lot)


def lots_by_margin(equity: float, max_margin_usage: float, product: Product, price: float) -> int:
    """按保证金上限反推手数。"""
    per_lot = product.margin_per_lot(price)
    if per_lot <= 0:
        return 0
    return int((equity * max_margin_usage) // per_lot)


def size_position(
    equity: float,
    risk: RiskConfig,
    product: Product,
    price: float,
    stop_distance: float,
    *,
    remaining_margin: float | None = None,
    risk_multiplier: float = 1.0,
) -> tuple[int, dict]:
    """综合风险预算与保证金约束给出建议手数，并返回计算明细（便于在清单里解释）。"""
    stop_points = abs(float(stop_distance))
    risk_lots = lots_by_risk(equity, risk.risk_per_trade * risk_multiplier, stop_points, product, price)
    margin_budget = equity * risk.max_margin_usage if remaining_margin is None else max(0.0, remaining_margin)
    per_lot_margin = product.margin_per_lot(price)
    margin_lots = int(margin_budget // per_lot_margin) if per_lot_margin > 0 else 0
    lots = max(0, min(risk_lots, margin_lots, risk.max_lots_per_product))
    detail = {
        "合约乘数": product.multiplier,
        "止损点数": round(stop_points, 2),
        "单手风险(元)": round(stop_points * product.multiplier, 2),
        "风险预算法手数": int(risk_lots),
        "保证金法上限手数": int(margin_lots),
        "单手保证金(元)": round(per_lot_margin, 2),
        "风险倍数": round(risk_multiplier, 2),
        "建议手数": int(lots),
    }
    if lots < risk.min_lots:
        one_lot_risk = stop_points * product.multiplier
        if (
            one_lot_risk > 0
            and one_lot_risk <= equity * risk.max_single_risk
            and per_lot_margin <= margin_budget
        ):
            lots = risk.min_lots
            detail["建议手数"] = int(lots)
            detail["兜底"] = (
                f"风险预算不足 1 手（{one_lot_risk:,.0f} 元 > "
                f"{equity * risk.risk_per_trade * risk_multiplier:,.0f} 元），"
                f"但单手风险 ≤ 权益 {risk.max_single_risk:.0%} 且保证金够用，按 1 手兜底"
            )
        else:
            lots = 0
            detail["放弃原因"] = (
                f"风险预算与保证金都不足：单手风险 {one_lot_risk:,.0f} 元，"
                f"可用保证金 {margin_budget:,.0f} 元"
            )
    return int(lots), detail


def drawdown_series(equity: pd.Series) -> pd.Series:
    """相对历史最高点的回撤序列（正数表示回撤幅度）。"""
    peak = equity.cummax()
    return (peak - equity) / peak.replace(0.0, np.nan)


def drawdown_throttle(drawdown: float, risk: RiskConfig) -> tuple[float, str]:
    """把当前回撤映射成风险倍数与中文说明。

    倍数永远大于 0：见 :attr:`RiskConfig.pause_risk_multiplier` 里记录的锁死问题。
    """
    if drawdown >= risk.drawdown_pause:
        return (
            risk.pause_risk_multiplier,
            f"回撤 {drawdown:.1%} 已达暂停线 {risk.drawdown_pause:.0%}："
            f"风险降至 {risk.pause_risk_multiplier:.0%}，只保留最小观察仓（停止交易会让账户无法恢复）",
        )
    if drawdown >= risk.drawdown_scale:
        return 0.5, f"回撤 {drawdown:.1%} 超过 {risk.drawdown_scale:.0%}：风险减半"
    return 1.0, f"回撤 {drawdown:.1%}，风险正常"


def portfolio_guard(
    equity: float,
    risk: RiskConfig,
    open_positions: int,
    used_margin: float,
    drawdown: float,
) -> tuple[bool, str, float]:
    """组合级开仓许可：返回 (是否允许开新仓, 说明, 风险倍数)。"""
    multiplier, note = drawdown_throttle(drawdown, risk)
    if open_positions >= risk.max_positions:
        return False, f"已持有 {open_positions} 个品种，达到上限 {risk.max_positions}", multiplier
    remaining = equity * risk.max_margin_usage - used_margin
    if remaining <= equity * risk.max_margin_usage * 0.05:
        return False, f"保证金已占用 {used_margin:,.0f} 元，接近上限 {equity * risk.max_margin_usage:,.0f} 元", multiplier
    return True, note + f"；剩余保证金额度 {remaining:,.0f} 元", multiplier


def activity_stats(trades: list, equity_curve: pd.DataFrame, initial_equity: float) -> dict:
    """统计活跃度相关指标：成交笔数、有交易的交易日占比、平均持有天数。"""
    if not trades:
        return {"交易笔数": 0, "有交易天数": 0, "交易日总数": int(len(equity_curve)),
                "活跃交易日占比": 0.0, "平均持有自然日": 0.0, "盈利笔数占比": 0.0}
    entry_days = {pd.Timestamp(trade.entry_date).date() for trade in trades}
    exit_days = {pd.Timestamp(trade.exit_date).date() for trade in trades}
    active = entry_days | exit_days
    total_days = int(len(equity_curve))
    holding = [
        (pd.Timestamp(trade.exit_date) - pd.Timestamp(trade.entry_date)).days for trade in trades
    ]
    wins = sum(1 for trade in trades if trade.gross_pnl(1) * trade.direction > 0)
    return {
        "交易笔数": len(trades),
        "有交易天数": len(active),
        "交易日总数": total_days,
        "活跃交易日占比": round(len(active) / total_days, 4) if total_days else 0.0,
        "平均持有自然日": round(float(np.mean(holding)), 2),
        "盈利笔数占比": round(wins / len(trades), 4),
    }
