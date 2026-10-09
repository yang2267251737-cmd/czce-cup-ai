"""策略层：把特征表变成"每一天应该持有什么方向"的状态序列，并给出可挂单的具体价位。

两条互补策略
- :func:`breakout_positions` —— 唐奇安通道突破（海龟式），负责**收益**。它天然是低频率的。
- :func:`meanrev_positions` —— 布林带均值回归（只在震荡市开仓），负责**活跃度**。

执行假设（回测与实盘计划共用同一套，避免"回测赚钱、实盘不知道挂哪"）
1. 入场：用**上一根收盘**判断，在**下一根开盘**成交。
2. 止损：进场后立即挂保护性止损，盘中触及即成交；跳空穿越则按更差的开盘价成交。
3. 离场：反向通道/均值回归目标用上一根收盘判断，下一根开盘成交。
4. 移动止损（吊灯止损）用当日收盘更新，只朝有利方向移动。

性能说明：状态机里**不逐行取 ``DataFrame.iloc[i]``**（5000 根 × 数十组参数会慢到分钟级），
而是先把需要的列取成 numpy 数组再按下标访问，单次全历史回测从秒级降到几十毫秒。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_FEATURES = (
    "date", "open", "high", "low", "close", "atr",
    "donchian_up", "donchian_dn", "exit_up", "exit_dn", "ema_slow", "adx", "boll_z",
)


@dataclass
class StrategyParams:
    """策略参数。默认值来自海龟法则的常用设置，未做参数优化。"""

    entry_window: int = 20
    exit_window: int = 10
    atr_window: int = 14
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    use_trend_filter: bool = True
    adx_min: float = 0.0
    allow_short: bool = True
    max_hold_days: int = 0
    entry_z: float = 2.0
    exit_z: float = 0.0
    regime_max_adx: float = 20.0

    def label(self) -> str:
        """给报告用的可读名称。"""
        return (f"N={self.entry_window}/M={self.exit_window}/ATR{self.atr_window}"
                f"/止损{self.stop_atr}ATR/跟踪{self.trail_atr}ATR"
                f"{'/趋势过滤' if self.use_trend_filter else ''}")


@dataclass
class Trade:
    """一笔完整的开平记录，单位是"1 手"。"""

    code: str
    direction: int
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    lots: float = 1.0
    exit_reason: str = ""
    entry_reason: str = ""
    initial_stop: float = float("nan")
    """建仓时挂出的初始止损价，用来算这笔交易承担了多少风险（R 的基数）。"""

    def risk_per_lot(self, multiplier: int) -> float:
        """建仓时单手的风险金额（元）= 入场价与初始止损的距离 × 合约乘数。"""
        if not np.isfinite(self.initial_stop):
            return float("nan")
        return abs(self.entry_price - self.initial_stop) * multiplier

    def r_multiple(self, multiplier: int) -> float:
        """这笔交易的 R 倍数：毛盈亏 ÷ 建仓风险。趋势策略通常用 R 而非金额来评价。"""
        risk = self.risk_per_lot(multiplier)
        if not np.isfinite(risk) or risk <= 0:
            return float("nan")
        return self.gross_pnl(multiplier) / risk

    def gross_pnl(self, multiplier: int) -> float:
        """未扣成本的毛盈亏（元）。"""
        return (self.exit_price - self.entry_price) * self.direction * multiplier * self.lots

    def to_row(self, product) -> dict:
        """转成便于表格展示的一行。"""
        days = int(np.busday_count(pd.Timestamp(self.entry_date).date(), pd.Timestamp(self.exit_date).date()))
        return {
            "品种": product.code,
            "方向": "多" if self.direction > 0 else "空",
            "手数": self.lots,
            "开仓日": pd.Timestamp(self.entry_date).date(),
            "开仓价": round(self.entry_price, 2),
            "平仓日": pd.Timestamp(self.exit_date).date(),
            "平仓价": round(self.exit_price, 2),
            "持有天": days,
            "毛盈亏": round(self.gross_pnl(product.multiplier), 2),
            "R倍数": round(self.r_multiple(product.multiplier), 2),
            "离场原因": self.exit_reason,
            "开仓原因": self.entry_reason,
        }


@dataclass
class PositionPlan:
    """下一根 bar 的执行方案，供每日操作清单直接使用。"""

    code: str
    action: str
    direction: int
    reason: str
    entry_trigger: float = float("nan")
    stop: float = float("nan")
    exit_trigger: float = float("nan")
    note: str = ""
    extras: dict = field(default_factory=dict)


def as_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """把特征表转成列式 numpy 数组，供状态机快速按下标访问。"""
    missing = [name for name in REQUIRED_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"特征表缺少列: {missing}")
    return {name: frame[name].to_numpy() for name in REQUIRED_FEATURES}


def _entry_conditions(bars: dict[str, np.ndarray], i: int, params: StrategyParams) -> tuple[bool, bool, str]:
    """在**第 i 根**（即决策依据的那根收盘）判断多空入场条件。"""
    close = float(bars["close"][i])
    up = float(bars["donchian_up"][i])
    if not np.isfinite(up) or not np.isfinite(bars["atr"][i]):
        return False, False, "特征未预热"
    long_ok = close > up
    short_ok = close < float(bars["donchian_dn"][i])
    if params.use_trend_filter:
        slow = float(bars["ema_slow"][i])
        long_ok = long_ok and close > slow
        short_ok = short_ok and close < slow
    if params.adx_min > 0:
        adx_value = float(bars["adx"][i])
        long_ok = long_ok and adx_value >= params.adx_min
        short_ok = short_ok and adx_value >= params.adx_min
    if not params.allow_short:
        short_ok = False
    if long_ok:
        return True, False, f"收盘突破 {params.entry_window} 日高点"
    if short_ok:
        return False, True, f"收盘跌破 {params.entry_window} 日低点"
    return False, False, ""


def _exit_condition(bars: dict[str, np.ndarray], i: int, direction: int, params: StrategyParams) -> tuple[bool, str]:
    """在第 i 根收盘判断趋势离场条件。"""
    close = float(bars["close"][i])
    if direction > 0:
        level = float(bars["exit_dn"][i])
        if np.isfinite(level) and close < level:
            return True, f"收盘跌破 {params.exit_window} 日低点"
    else:
        level = float(bars["exit_up"][i])
        if np.isfinite(level) and close > level:
            return True, f"收盘突破 {params.exit_window} 日高点"
    return False, ""


def _meanrev_entry(bars: dict[str, np.ndarray], i: int, params: StrategyParams) -> tuple[bool, bool, str]:
    """震荡市里的均值回归入场：布林带 z 分数极端 + ADX 不过高。"""
    z = float(bars["boll_z"][i])
    adx_value = float(bars["adx"][i])
    if not np.isfinite(z) or not np.isfinite(adx_value) or not np.isfinite(bars["atr"][i]):
        return False, False, "特征未预热"
    if adx_value > params.regime_max_adx:
        return False, False, ""
    if z <= -params.entry_z:
        return True, False, f"布林 z={z:.2f} 超卖且 ADX={adx_value:.0f} 偏低"
    if z >= params.entry_z and params.allow_short:
        return False, True, f"布林 z={z:.2f} 超买且 ADX={adx_value:.0f} 偏低"
    return False, False, ""


def _meanrev_exit(bars: dict[str, np.ndarray], i: int, direction: int, params: StrategyParams) -> tuple[bool, str]:
    """回归到中轨（z 回到 exit_z）即离场。"""
    z = float(bars["boll_z"][i])
    if not np.isfinite(z):
        return False, ""
    if direction > 0 and z >= params.exit_z:
        return True, f"z 回到 {z:.2f}，均值回归完成"
    if direction < 0 and z <= params.exit_z:
        return True, f"z 回到 {z:.2f}，均值回归完成"
    return False, ""


def _state_machine(
    features: pd.DataFrame,
    params: StrategyParams,
    entry_fn,
    exit_fn,
    code: str,
    max_hold_days: int,
) -> tuple[pd.DataFrame, list[Trade], list[list[tuple]]]:
    """通用的"收盘决策 + 次日开盘成交 + 盘中挂止损"状态机，回测与实盘计划共用。

    返回值第三项是**逐 bar 的成交事件**（``events[i]``），组合回测用它来按真实手数结算，
    这样单品种信号逻辑只需要维护一份，避免"回测一套、实盘计划另一套"的口径分裂。
    每个事件的格式：``("exit", 成交价, 原因)`` 或 ``("entry", 方向, 成交价, 原因, 初始止损)``。
    """
    frame = features.reset_index(drop=True)
    n = len(frame)
    if n == 0:
        return pd.DataFrame(), [], []
    bars = as_arrays(frame)
    dates = bars["date"]
    opens, highs, lows, closes = bars["open"], bars["high"], bars["low"], bars["close"]
    atrs = bars["atr"]

    target_state = np.zeros(n, dtype=int)
    stop_series = np.full(n, np.nan)
    current = 0
    stop_price = np.nan
    entry_price = np.nan
    entry_idx = -1
    entry_note = ""
    initial_stop = np.nan
    trades: list[Trade] = []
    events: list[list[tuple]] = [[] for _ in range(n)]

    def close_trade(index: int, price: float, reason: str) -> None:
        trades.append(
            Trade(
                code=code,
                direction=current,
                entry_date=dates[entry_idx],
                entry_price=float(entry_price),
                exit_date=dates[index],
                exit_price=float(price),
                exit_reason=reason,
                entry_reason=entry_note,
                initial_stop=float(initial_stop),
            )
        )

    for i in range(n):
        open_price = float(opens[i])
        high, low, close = float(highs[i]), float(lows[i]), float(closes[i])
        atr_now = float(atrs[i]) if np.isfinite(atrs[i]) else np.nan

        # A. 已持仓：先看挂着的保护性止损是否在盘中被打到
        if current != 0 and np.isfinite(stop_price):
            hit = (current > 0 and low <= stop_price) or (current < 0 and high >= stop_price)
            if hit:
                # 跳空穿越止损价时按更差的开盘价成交，避免高估回测收益
                fill = min(open_price, stop_price) if current > 0 else max(open_price, stop_price)
                events[i].append(("exit", float(fill), "止损"))
                close_trade(i, fill, "止损")
                current, stop_price, entry_price, entry_idx, initial_stop = 0, np.nan, np.nan, -1, np.nan

        # B. 已持仓：上一根收盘给出的离场信号，今日开盘执行
        if current != 0 and i >= 1:
            should_exit, reason = exit_fn(bars, i - 1, current, params)
            if max_hold_days and (i - entry_idx) >= max_hold_days:
                should_exit, reason = True, f"持有满 {max_hold_days} 日"
            if should_exit:
                events[i].append(("exit", open_price, reason))
                close_trade(i, open_price, reason)
                current, stop_price, entry_price, entry_idx, initial_stop = 0, np.nan, np.nan, -1, np.nan

        # C. 空仓：上一根收盘给出的入场信号，今日开盘执行
        if current == 0 and i >= 1 and np.isfinite(atr_now):
            want_long, want_short, note = entry_fn(bars, i - 1, params)
            direction = 1 if want_long else (-1 if want_short else 0)
            if direction != 0:
                current, entry_price, entry_idx, entry_note = direction, open_price, i, note
                stop_price = entry_price - direction * params.stop_atr * atr_now
                initial_stop = stop_price
                events[i].append(("entry", direction, open_price, note, float(stop_price)))
                # 入场当日即被打止损
                if (direction > 0 and low <= stop_price) or (direction < 0 and high >= stop_price):
                    fill = min(open_price, stop_price) if direction > 0 else max(open_price, stop_price)
                    events[i].append(("exit", float(fill), "入场当日止损"))
                    close_trade(i, fill, "入场当日止损")
                    current, stop_price, entry_price, entry_idx, initial_stop = 0, np.nan, np.nan, -1, np.nan

        # D. 持仓中：用当日收盘更新移动止损，只朝有利方向移动
        if current != 0 and np.isfinite(atr_now):
            candidate = close - current * params.trail_atr * atr_now
            stop_price = candidate if not np.isfinite(stop_price) else (
                max(stop_price, candidate) if current > 0 else min(stop_price, candidate)
            )

        target_state[i] = current
        stop_series[i] = stop_price

    out = pd.DataFrame(
        {
            "date": dates,
            "close": closes,
            "target_state": target_state,
            "stop": stop_series,
        }
    )
    return out, trades, events


def breakout_positions(
    features: pd.DataFrame, params: StrategyParams, code: str = ""
) -> tuple[pd.DataFrame, list[Trade], list[list[tuple]]]:
    """唐奇安突破策略：收益主引擎。"""
    return _state_machine(features, params, _entry_conditions, _exit_condition, code, params.max_hold_days)


def meanrev_positions(
    features: pd.DataFrame, params: StrategyParams, code: str = ""
) -> tuple[pd.DataFrame, list[Trade], list[list[tuple]]]:
    """布林带均值回归策略：活跃度补充引擎，默认最多持有 10 个交易日。"""
    return _state_machine(features, params, _meanrev_entry, _meanrev_exit, code, params.max_hold_days or 10)


def next_bar_plan(
    features: pd.DataFrame,
    params: StrategyParams,
    positions: pd.DataFrame,
    code: str,
) -> PositionPlan:
    """根据最新一根已收盘 bar，给出**下一根 bar** 可直接挂单的执行方案。

    返回的价位就是字面意义上的挂单参考：
    - 空仓：向上突破 ``entry_trigger`` 做多、向下跌破另一侧做空；
    - 持仓：``stop`` 是必须挂出的止损价，``exit_trigger`` 是收盘跌破就要走的位置。
    """
    frame = features.dropna(subset=["atr"]).reset_index(drop=True)
    if frame.empty or positions.empty:
        return PositionPlan(code, "等待", 0, "特征未预热")
    last = frame.iloc[-1]
    state = int(positions["target_state"].iloc[-1])
    atr_now = float(last["atr"])
    close = float(last["close"])

    if state != 0:
        stop = float(positions["stop"].iloc[-1])
        exit_ref = float(last["exit_dn"] if state > 0 else last["exit_up"])
        exit_gap = close - exit_ref if state > 0 else exit_ref - close
        return PositionPlan(
            code=code,
            action="持有",
            direction=state,
            reason=f"处于{'多' if state > 0 else '空'}头，移动止损 {stop:.1f}",
            stop=round(stop, 2),
            exit_trigger=round(exit_ref, 2),
            note=(f"收盘距反向离场位 {exit_gap:.1f} 点（{exit_gap / close:.2%}）；"
                  f"收盘跌破/突破即次日开盘离场"),
        )

    window = int(params.entry_window)
    up_trigger = float(frame["high"].tail(window).max())
    dn_trigger = float(frame["low"].tail(window).min())
    return PositionPlan(
        code=code,
        action="观望",
        direction=0,
        reason=f"未突破 {window} 日通道",
        entry_trigger=round(up_trigger, 2),
        stop=round(dn_trigger, 2),
        note=(f"向上突破 {up_trigger:.1f} 做多（还需 {up_trigger - close:.1f} 点 / "
              f"{(up_trigger - close) / close:.2%}）；向下跌破 {dn_trigger:.1f} 做空"
              f"（还需 {close - dn_trigger:.1f} 点 / {(close - dn_trigger) / close:.2%}）；"
              f"当日 ATR≈{atr_now:.1f}"),
    )


def combine_states(primary: pd.DataFrame, satellite: pd.DataFrame | None = None) -> pd.DataFrame:
    """把主策略与辅助策略的仓位合并；同向取主策略，冲突时以主策略为准（辅助让位）。"""
    merged = primary.copy()
    if satellite is None or satellite.empty:
        return merged
    sat = satellite.set_index("date")["target_state"].reindex(merged["date"]).fillna(0).astype(int).to_numpy()
    base = merged["target_state"].to_numpy()
    merged["satellite_state"] = sat
    merged["target_state"] = np.where(base != 0, base, sat)
    return merged
