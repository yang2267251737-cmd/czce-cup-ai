"""组合回测层：把单品种信号接到真实手数、保证金、1.5 倍手续费和滑点上。

口径说明（这些选择会影响结论，所以写清楚）
1. **手续费**：开仓/平仓各收一次，按赛制 **1.5 倍**交易所标准；当日开当日平按"平今"费率。
2. **滑点**：每次成交额外按 1 个最小变动价位计（保守），买卖都算。
3. **保证金**：按交易所标准 × 1.3 安全垫占用，超过上限就不许再开新仓。
4. **成交时点**：信号在收盘产生、次日开盘成交；止损是盘中挂单，触及即成交。
5. **不加杠杆放大**：手数由"单笔风险 1% + 保证金上限"共同决定，不做满仓。

已知偏差：新浪连续合约在换月处有跳空，回测会因此产生噪声；真实交易需按主力合约代码下单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from contracts import Product
from risk import RiskConfig, drawdown_series, portfolio_guard, size_position
from strategy import StrategyParams

TRADING_DAYS_PER_YEAR = 244


@dataclass
class OpenPosition:
    """组合里的一条持仓。"""

    code: str
    direction: int
    lots: int
    entry_price: float
    entry_date: pd.Timestamp
    stop: float
    entry_reason: str = ""


@dataclass
class BacktestResult:
    """回测结果容器，直接可序列化成报告。"""

    name: str
    equity_curve: pd.DataFrame
    trades: list[dict]
    events: list[dict]
    metrics: dict[str, Any] = field(default_factory=dict)
    params_note: str = ""
    final_positions: list[dict[str, Any]] = field(default_factory=list)
    """回测结束时的未平仓持仓，供每日计划把"当前该拿什么仓"对出来。"""
    initial_equity: float = 0.0

    def summary_lines(self) -> list[str]:
        """给 Markdown 报告用的中文摘要行。"""
        lines = [f"- 策略：{self.name}", f"- 参数：{self.params_note}"]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                if "率" in key or "比" in key and key not in {"盈亏比"}:
                    lines.append(f"- {key}：{value:.2%}" if abs(value) <= 5 else f"- {key}：{value:.2f}")
                else:
                    lines.append(f"- {key}：{value:,.2f}")
            else:
                lines.append(f"- {key}：{value}")
        return lines


def _mark_to_market(positions: dict[str, OpenPosition], close_prices: dict[str, float], products: dict[str, Product]) -> float:
    """按当日收盘价计算持仓浮动盈亏（元）。"""
    total = 0.0
    for code, position in positions.items():
        price = close_prices.get(code)
        if price is None or not np.isfinite(price):
            continue
        total += (
            (price - position.entry_price) * position.direction * products[code].multiplier * position.lots
        )
    return total


def _used_margin(positions: dict[str, OpenPosition], close_prices: dict[str, float], products: dict[str, Product]) -> float:
    """占用保证金合计（元）。"""
    total = 0.0
    for code, position in positions.items():
        price = close_prices.get(code, position.entry_price)
        total += products[code].margin_per_lot(float(price)) * position.lots
    return total


def prepare_signals(
    frames: dict[str, pd.DataFrame],
    products: dict[str, Product],
    params: StrategyParams,
    satellite: StrategyParams | None = None,
) -> dict[str, dict[str, Any]]:
    """对每个品种跑一遍状态机，得到逐日事件表（组合层只负责资金与手数）。"""
    from indicators import compute_features
    from strategy import breakout_positions, combine_states, meanrev_positions
    from universe import code_of

    prepared: dict[str, dict[str, Any]] = {}
    for code, raw in frames.items():
        product_code = code_of(code)
        if product_code not in products:
            raise KeyError(f"品种池里的 {code} 在合约规格快照中找不到（{product_code}），请先运行 scripts/contracts.py")
        features = compute_features(raw, {"atr": params.atr_window, "fast": params.entry_window,
                                          "slow": 50, "donchian": params.entry_window,
                                          "exit": params.exit_window, "adx": 14, "boll": 20})
        primary, trades, events = breakout_positions(features, params, code)
        sat_trades: list = []
        if satellite is not None:
            sat_states, sat_trades, sat_events = meanrev_positions(features, satellite, code)
            combined = combine_states(primary, sat_states)
            # 辅助策略的事件只在主策略空仓时采纳，避免两条腿在同一品种上互相打架
            primary_codes = primary["target_state"].to_numpy()
            merged_events: list[list[tuple]] = []
            for index, (main_events, sat_event_list) in enumerate(zip(events, sat_events)):
                if main_events:
                    merged_events.append(main_events)
                elif primary_codes[index] == 0 and sat_event_list:
                    merged_events.append(sat_event_list)
                else:
                    merged_events.append([])
            primary = combined
            events = merged_events
        prepared[code] = {
            "features": features,
            "states": primary,
            "events": events,
            "trades": trades,
            "satellite_trades": sat_trades,
            "product": products[product_code],
            "product_code": product_code,
        }
    return prepared


def run_backtest(
    prepared: dict[str, dict[str, Any]],
    risk: RiskConfig,
    *,
    name: str = "组合回测",
    params_note: str = "",
    start: str | None = None,
    end: str | None = None,
    verbose: bool = False,
) -> BacktestResult:
    """按事件驱动跑组合回测，返回净值曲线、成交明细与指标。"""
    date_sets = [
        set(pd.to_datetime(item["states"]["date"]).dt.normalize())
        for item in prepared.values()
    ]
    all_dates = sorted(set.union(*date_sets)) if date_sets else []
    if start:
        all_dates = [day for day in all_dates if day >= pd.Timestamp(start)]
    if end:
        all_dates = [day for day in all_dates if day <= pd.Timestamp(end)]
    if not all_dates:
        raise ValueError("回测区间内没有交易日")

    lookup: dict[str, dict[pd.Timestamp, int]] = {}
    for code, item in prepared.items():
        normalized = pd.to_datetime(item["states"]["date"]).dt.normalize()
        lookup[code] = {day: index for index, day in enumerate(normalized)}

    initial_equity = risk.equity
    realized = 0.0
    positions: dict[str, OpenPosition] = {}
    rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    peak_equity = initial_equity
    total_fees = 0.0
    skipped = 0

    def equity_now() -> float:
        """账户权益 = 初始权益 + 累计已实现盈亏 + 当前浮动盈亏。

        **这里曾经写错过**：只算了初始权益 + 浮动盈亏，漏掉累计已实现盈亏，
        于是账户一旦浮盈回吐就被误判为"回撤超标"，风险倍数长期卡在 0.25，
        导致绝大多数信号连 1 手都开不出来（实测 138 笔成交 / 518 次被拒）。
        权益口径是本文件里最容易错、后果最大的地方，改动时必须先跑冒烟测试。
        """
        floating = _mark_to_market(positions, close_prices, product_of)
        return initial_equity + realized + floating

    for day in all_dates:
        close_prices: dict[str, float] = {}
        for code, item in prepared.items():
            index = lookup[code].get(day)
            if index is None:
                continue
            close_prices[code] = float(item["features"].at[index, "close"])
        product_of = {code: prepared[code]["product"] for code in positions}

        # ---- 1) 先处理离场（止损或开盘离场），结算盈亏与手续费 ----
        for code, item in prepared.items():
            index = lookup[code].get(day)
            if index is None:
                continue
            for event in item["events"][index]:
                if event[0] != "exit":
                    continue
                position = positions.pop(code, None)
                if position is None:
                    continue
                fill_price = float(event[1])
                reason = event[2]
                product = item["product"]
                gross = (fill_price - position.entry_price) * position.direction * product.multiplier * position.lots
                close_today = pd.Timestamp(position.entry_date).normalize() == pd.Timestamp(day).normalize()
                fee = product.charged_fee(fill_price, position.lots, opening=False, close_today=close_today)
                slip = product.slippage(position.lots)
                realized += gross - fee - slip
                total_fees += fee
                trade_rows.append(
                    {
                        "品种": code,
                        "方向": "多" if position.direction > 0 else "空",
                        "手数": position.lots,
                        "开仓日": pd.Timestamp(position.entry_date).date(),
                        "开仓价": round(position.entry_price, 2),
                        "平仓日": pd.Timestamp(day).date(),
                        "平仓价": round(fill_price, 2),
                        "持有天数": int(np.busday_count(pd.Timestamp(position.entry_date).date(), pd.Timestamp(day).date())),
                        "毛盈亏": round(gross, 2),
                        "手续费": round(fee, 2),
                        "净盈亏": round(gross - fee - slip, 2),
                        "离场原因": reason,
                        "日内平今": close_today,
                    }
                )
                event_rows.append({"日期": pd.Timestamp(day).date(), "品种": code, "动作": "平仓",
                                   "方向": "多" if position.direction > 0 else "空",
                                   "手数": position.lots, "成交价": round(fill_price, 2), "原因": reason})
                product_of = {code_: prepared[code_]["product"] for code_ in positions}

        # ---- 2) 再处理入场，手数由当前权益与回撤状态决定 ----
        for code, item in prepared.items():
            index = lookup[code].get(day)
            if index is None or code in positions:
                continue
            product = item["product"]
            for event in item["events"][index]:
                if event[0] != "entry" or code in positions:
                    continue
                direction, fill_price, reason, stop = event[1], float(event[2]), event[3], float(event[4])
                current_equity = equity_now()
                peak_equity = max(peak_equity, current_equity)
                drawdown = 0.0 if peak_equity <= 0 else (peak_equity - current_equity) / peak_equity
                allowed, note, risk_multiplier = portfolio_guard(
                    current_equity, risk, len(positions), _used_margin(positions, close_prices, product_of), drawdown
                )
                if not allowed:
                    skipped += 1
                    continue
                stop_distance = abs(fill_price - stop)
                lots, detail = size_position(
                    current_equity, risk, product, fill_price, stop_distance, risk_multiplier=risk_multiplier
                )
                if lots <= 0:
                    skipped += 1
                    continue
                fee = product.charged_fee(fill_price, lots, opening=True)
                slip = product.slippage(lots)
                realized -= fee + slip
                total_fees += fee
                positions[code] = OpenPosition(code, direction, lots, fill_price, pd.Timestamp(day), stop, reason)
                product_of = {code_: prepared[code_]["product"] for code_ in positions}
                event_rows.append({"日期": pd.Timestamp(day).date(), "品种": code, "动作": "开仓",
                                   "方向": "多" if direction > 0 else "空", "手数": lots,
                                   "成交价": round(fill_price, 2), "原因": reason})
                if verbose:
                    print(f"{day.date()} 开仓 {code} {'多' if direction > 0 else '空'} {lots} 手 @{fill_price:.1f}；{detail}")

        # ---- 3) 收盘盯市，记录净值 ----
        floating = _mark_to_market(positions, close_prices, product_of)
        total_equity = equity_now()
        peak_equity = max(peak_equity, total_equity)
        rows.append(
            {
                "date": day,
                "equity": total_equity,
                "realized": realized,
                "floating": floating,
                "positions": len(positions),
                "margin": _used_margin(positions, close_prices, product_of),
                "fees_cum": total_fees,
            }
        )

    curve = pd.DataFrame(rows)
    metrics = compute_metrics(curve, trade_rows, risk.equity, total_fees, skipped)
    final_positions = [
        {
            "code": position.code,
            "direction": int(position.direction),
            "lots": int(position.lots),
            "entry_price": round(float(position.entry_price), 2),
            "entry_date": str(pd.Timestamp(position.entry_date).date()),
            "stop": round(float(position.stop), 2) if np.isfinite(position.stop) else None,
            "last_price": round(float(close_prices.get(position.code, position.entry_price)), 2),
            "entry_reason": position.entry_reason,
        }
        for position in positions.values()
    ]
    return BacktestResult(
        name=name,
        equity_curve=curve,
        trades=trade_rows,
        events=event_rows,
        metrics=metrics,
        params_note=params_note,
        final_positions=final_positions,
        initial_equity=risk.equity,
    )


def compute_metrics(
    curve: pd.DataFrame, trades: list[dict], initial_equity: float, total_fees: float, skipped: int
) -> dict[str, Any]:
    """从净值曲线与成交明细算出赛制关心的全部指标。"""
    if curve.empty:
        return {}
    equity = curve["equity"]
    final = float(equity.iloc[-1])
    total_return = final / initial_equity - 1
    days = max(1, len(curve))
    years = days / TRADING_DAYS_PER_YEAR
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else float("nan")
    daily_returns = equity.pct_change().dropna()
    volatility = float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_returns) > 1 else 0.0
    sharpe = float(daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0 else 0.0
    drawdown = drawdown_series(equity)
    max_dd = float(drawdown.max())
    calmar = float(annualized / max_dd) if max_dd > 0 and np.isfinite(annualized) else float("nan")
    wins = [row for row in trades if row["净盈亏"] > 0]
    losses = [row for row in trades if row["净盈亏"] <= 0]
    gross_win = sum(row["净盈亏"] for row in wins)
    gross_loss = abs(sum(row["净盈亏"] for row in losses))
    activity = activity_stats_from_rows(trades, curve)
    return {
        "期末权益": round(final, 2),
        "总收益率": round(total_return, 6),
        "年化收益率": round(annualized, 6) if np.isfinite(annualized) else float("nan"),
        "最大回撤": round(max_dd, 6),
        "年化波动率": round(volatility, 6),
        "夏普比率": round(sharpe, 4),
        "卡玛比率": round(calmar, 4) if np.isfinite(calmar) else float("nan"),
        "交易笔数": len(trades),
        "胜率": round(len(wins) / len(trades), 4) if trades else 0.0,
        "盈亏比": round(gross_win / gross_loss, 4) if gross_loss > 0 else float("inf"),
        "手续费合计": round(total_fees, 2),
        "手续费占初始权益": round(total_fees / initial_equity, 6),
        "因风控放弃的开仓次数": skipped,
        **activity,
    }


def activity_stats_from_rows(trades: list[dict], curve: pd.DataFrame) -> dict[str, Any]:
    """活跃度指标：郑商所杯的活跃度得分与"有没有在交易"直接相关。"""
    if not trades:
        return {"有交易天数": 0, "交易日总数": int(len(curve)), "活跃交易日占比": 0.0,
                "平均持有交易日": 0.0}
    active = {row["开仓日"] for row in trades} | {row["平仓日"] for row in trades}
    total = int(len(curve))
    return {
        "有交易天数": len(active),
        "交易日总数": total,
        "活跃交易日占比": round(len(active) / total, 4) if total else 0.0,
        "平均持有交易日": round(float(np.mean([row["持有天数"] for row in trades])), 2),
    }


def curve_to_frame(result: BacktestResult) -> pd.DataFrame:
    """把净值曲线整理成带日期的表，便于画图和写报告。"""
    curve = result.equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve["drawdown"] = drawdown_series(curve["equity"])
    return curve


def compare_results(results: list[BacktestResult]) -> pd.DataFrame:
    """把多个回测结果并排比成一张表。"""
    keys = ["总收益率", "年化收益率", "最大回撤", "夏普比率", "卡玛比率", "交易笔数",
            "胜率", "盈亏比", "手续费合计", "活跃交易日占比"]
    rows = []
    for result in results:
        row = {"策略": result.name}
        for key in keys:
            row[key] = result.metrics.get(key, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)
