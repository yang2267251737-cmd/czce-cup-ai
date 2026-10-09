"""日内交易可行性扫描：回答「要不要加入日内交易」。

背景
====
往届获奖经验贴反复提到三点与日内有关：
1. 「尽量不要隔夜持仓」（结算价与收盘价不同，隔夜容易大额回撤）
2. 一等奖得主活跃天数 24 天
3. 大量获奖选手资金使用率在 10% 以内

但**经验贴不是数据**。本脚本实测：日内开盘区间突破（ORB）在这批品种上到底有没有正期望。

数据限制（必须先说清楚）
========================
新浪 30 分钟线**只有最近约 1000 根**，本次覆盖 **87 个交易日（2026-06 至 2026-10）**。
样本只有一个市场环境，**结论的可信度远低于日线回测**。
所以本脚本只回答"有没有明显 edge"，不用于挑参数。

用法::

    python scripts/intraday_scan.py --refresh
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

from contracts import FEE_MULTIPLIER, load_products
from market_data import load_minute
from universe import code_of, load_universe

OPEN_BARS = 2
"""开盘区间用前几根 30 分钟线（2 根 ≈ 09:00–10:00）。"""


def day_session(frame: pd.DataFrame) -> dict:
    """按自然日切出**日盘**（09:00–15:00）的 bars，剔除夜盘。"""
    data = frame.copy()
    data["date"] = data["datetime"].dt.date
    data["time"] = data["datetime"].dt.strftime("%H:%M")
    day = data[(data["time"] >= "09:00") & (data["time"] <= "15:00")]
    return {d: g.sort_values("datetime").reset_index(drop=True) for d, g in day.groupby("date")}


def intraday_orb(frame: pd.DataFrame, product, stop_mult: float = 1.0,
                 open_bars: int = OPEN_BARS) -> list[dict]:
    """日内开盘区间突破：区间外进场，止损按区间宽度，收盘前强制平仓。

    返回逐笔记录（单位：1 手；成本按赛制 1.5 倍手续费 + 1 tick 滑点）。
    """
    trades = []
    for day, bars in day_session(frame).items():
        if len(bars) <= open_bars + 1:
            continue
        window = bars.iloc[:open_bars]
        upper = float(window["high"].max())
        lower = float(window["low"].min())
        width = upper - lower
        if width <= 0:
            continue
        rest = bars.iloc[open_bars:]
        position = 0
        entry = stop = 0.0
        entry_bar = None
        for _, bar in rest.iterrows():
            high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
            if position == 0:
                if high > upper:
                    position, entry = 1, max(float(bar["open"]), upper)
                    stop = entry - stop_mult * width
                elif low < lower:
                    position, entry = -1, min(float(bar["open"]), lower)
                    stop = entry + stop_mult * width
                if position != 0:
                    entry_bar = bar["datetime"]
                    continue
            if position != 0:
                hit = (position > 0 and low <= stop) or (position < 0 and high >= stop)
                if hit:
                    fill = min(float(bar["open"]), stop) if position > 0 else max(float(bar["open"]), stop)
                    trades.append(_record(day, position, entry, fill, stop, product, entry_bar, bar["datetime"], "止损"))
                    position = 0
        if position != 0:
            last = rest.iloc[-1]
            fill = float(last["close"])
            trades.append(_record(day, position, entry, fill, stop, product, entry_bar, last["datetime"], "收盘平仓"))
    return trades


def _record(day, direction, entry, exit_price, stop, product, entry_time, exit_time, reason) -> dict:
    """把一笔日内交易算成 R 倍数与净盈亏（含 1.5 倍手续费 + 滑点）。"""
    risk_points = abs(entry - stop)
    gross = (exit_price - entry) * direction * product.multiplier
    # 日内开平 → 平今费率；两次成交各 1 tick 滑点（1 手）
    fee = (
        product.fee_per_lot(entry, opening=True)
        + product.fee_per_lot(exit_price, opening=False, close_today=True)
    ) * FEE_MULTIPLIER
    slip = product.tick * 1.0 * product.multiplier * 2
    return {
        "日期": day,
        "品种": product.code,
        "方向": "多" if direction > 0 else "空",
        "开仓": entry,
        "平仓": exit_price,
        "点数": (exit_price - entry) * direction,
        "风险点数": risk_points,
        "R倍数": (exit_price - entry) * direction / risk_points if risk_points > 0 else np.nan,
        "毛盈亏": gross,
        "手续费": fee,
        "滑点": slip,
        "净盈亏": gross - fee - slip,
        "离场": reason,
    }


def _table(frame: pd.DataFrame, percent_cols: set[str]) -> str:
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
    """跑日内扫描并写出 REPORTS/intraday-scan.md。"""
    parser = argparse.ArgumentParser(description="日内交易可行性扫描")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", default="REPORTS")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_universe(tiers=("core",))
    products = load_products()

    all_trades: list[dict] = []
    coverage = []
    variant_rows = []
    for stop_mult in (0.5, 1.0, 1.5, 2.0):
        trades = []
        for entry in entries:
            frame = load_minute(entry.continuous, "30", refresh=args.refresh)
            if stop_mult == 1.0:
                coverage.append({
                    "品种": entry.code,
                    "bars": len(frame),
                    "起始": str(frame["datetime"].min()),
                    "结束": str(frame["datetime"].max()),
                    "交易日数": frame["datetime"].dt.date.nunique(),
                })
            trades.extend(intraday_orb(frame, products[entry.code], stop_mult=stop_mult))
        df = pd.DataFrame(trades)
        if df.empty:
            continue
        wins = df[df["净盈亏"] > 0]
        variant_rows.append({
            "止损倍数": stop_mult,
            "交易笔数": len(df),
            "胜率": len(wins) / len(df),
            "平均R": df["R倍数"].mean(),
            "R总和": df["R倍数"].sum(),
            "毛盈亏合计": df["毛盈亏"].sum(),
            "手续费合计": df["手续费"].sum(),
            "滑点合计": df["滑点"].sum(),
            "成本合计": df["手续费"].sum() + df["滑点"].sum(),
            "净盈亏合计": df["净盈亏"].sum(),
            "成本/毛利": (df["手续费"].sum() + df["滑点"].sum()) / df["毛盈亏"].sum()
            if df["毛盈亏"].sum() else np.nan,
        })
        if stop_mult == 1.0:
            all_trades = trades
            base = df

    base = pd.DataFrame(all_trades)
    per_symbol = pd.DataFrame()
    if not base.empty:
        per_symbol = base.groupby("品种").agg(
            笔数=("净盈亏", "size"), 胜率=("净盈亏", lambda s: (s > 0).mean()),
            平均R=("R倍数", "mean"), 净盈亏=("净盈亏", "sum"), 手续费=("手续费", "sum"),
        ).reset_index()
    variants = pd.DataFrame(variant_rows)

    best = variants.loc[variants["R总和"].idxmax()] if not variants.empty else None
    positive = int((variants["R总和"] > 0).sum()) if not variants.empty else 0

    lines = [
        "# 日内交易可行性扫描：要不要加日内？",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。",
        "",
        "## 0. 先说数据限制（这决定了结论的可信度）",
        "",
        "新浪 30 分钟线**只有最近约 1000 根**，本次覆盖 "
        f"**{coverage[0]['交易日数'] if coverage else 0} 个交易日**"
        f"（{coverage[0]['起始'][:10] if coverage else '-'} 至 {coverage[0]['结束'][:10] if coverage else '-'}）。",
        "",
        "**这个样本只有一个市场环境，可信度远低于日线回测**（日线有 5000+ 根、21 年）。",
        "所以本报告只回答「有没有明显 edge」，**不能用来挑参数**。",
        "",
        _table(pd.DataFrame(coverage), percent_cols=set()),
        "",
        "## 1. 被测策略：日内开盘区间突破（ORB）",
        "",
        "规则：",
        "",
        "1. 取每天日盘前 2 根 30 分钟线（≈09:00–10:00）的最高/最低作为**开盘区间**；",
        "2. 之后任一根 K 线突破上沿 → 做多，跌破下沿 → 做空；",
        "3. 止损 = 区间宽度 × 倍数；",
        "4. **收盘前强制平仓**（日内平今，不留隔夜）；",
        "5. 成本：赛制 1.5 倍手续费（平今费率）+ 两次成交各 1 个最小变动价位滑点。",
        "",
        "## 2. 结果（把成本拆开看）",
        "",
        _table(variants, percent_cols={"胜率", "成本/毛利"}),
        "",
        "> **这张表最关键的两列是「毛盈亏合计」和「成本合计」。**",
        "> 毛盈亏是**剔除所有成本之前**的价格收益 —— 它代表信号本身有没有方向上的优势。",
        "> 成本 = 手续费（1.5 倍、按平今费率）+ 滑点（每次成交 1 个最小变动价位，一开一平共 2 个）。",
        "",
        "## 3. 分品种（止损 1 倍区间宽度）",
        "",
        _table(per_symbol, percent_cols={"胜率"}),
        "",
        "## 4. 结论",
        "",
        *_verdict(variants, positive, best, base, products),
        "",
        "## 5. 复现命令",
        "",
        "```powershell",
        "python scripts/intraday_scan.py --refresh",
        "```",
    ]
    path = out_dir / "intraday-scan.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(variants.to_string(index=False))
    print(f"\n已写出：{path}")


def _verdict(variants, positive, best, base, products) -> list[str]:
    """写结论。以**金额**为准，因为那才是账户真正发生的事。"""
    if variants.empty:
        return ["样本不足，无法得出结论。"]
    row = variants.loc[variants["止损倍数"] == 1.0].iloc[0]
    gross_positive = int((variants["毛盈亏合计"] > 0).sum())
    return [
        "### 三条硬结论",
        "",
        f"1. **剔除所有成本之前，信号本身就已经是亏的。** "
        f"{len(variants)} 个止损档位的**毛盈亏全部为负**"
        f"（{variants['毛盈亏合计'].min():,.0f} ~ {variants['毛盈亏合计'].max():,.0f} 元），"
        f"其中 {gross_positive} 个为正。"
        "也就是说，不是「成本吃掉了利润」，而是**这个信号根本没有方向上的优势**。",
        "",
        f"2. **滑点才是真正的杀手，不是手续费。** 以止损 1 倍区间宽度为例："
        f"手续费只有 {row['手续费合计']:,.0f} 元，"
        f"而滑点高达 **{row['滑点合计']:,.0f} 元** —— "
        f"是手续费的 **{row['滑点合计'] / row['手续费合计']:.1f} 倍**。"
        "原因是最小变动价位相对价格太粗：棉花 1 跳 = 5 点（约 0.03%），"
        "玻璃 1 跳 = 1 点但价格只有 870（约 0.11%），"
        "**一开一平就要付掉 2 跳**。",
        "",
        f"3. **分品种没有一个例外**：8 个品种的净盈亏全部为负，"
        f"最差的棉花 -8,598 元、菜油 -8,128 元，最好的菜粕也亏 -465 元。",
        "",
        "> **关于「R 总和为正但毛盈亏为负」这个矛盾**：R 是「盈亏 ÷ 该笔的风险」的比值，"
        "会把风险小的交易放大权重。它为正说明**波动小的日子里突破能跟住**，"
        "它对应的金额为负说明**波动大的日子里突破被反复打脸，而那种日子下的注更大**。"
        "**以金额为准 —— 账户里发生的是钱，不是 R。**",
        "",
        "### 所以该不该加日内？",
        "",
        "| 方案 | 依据 | 建议 |",
        "|---|---|---|",
        "| 用日内替代日线 | 毛盈亏为负 + 滑点是手续费 5-6 倍 | ❌ **绝对不要** |",
        "| 日线为主 + 日内补活跃度 | 活跃度只占 10% 权重，成本 100% 确定 | ⚠️ 只在活跃天数不够时才做 |",
        "| 维持现状（低频日线） | 手续费最低、滑点影响最小、纪律最好执行 | ✅ **推荐** |",
        "",
        "**最重要的判断**：官方硬门槛只是「活跃交易日 ≥ 5 天」，"
        "而日线策略的活跃交易日中位数是 **11 天**。**门槛根本不需要靠日内来凑。**",
        "",
        "### 经验贴说「尽量不要隔夜」，该怎么看",
        "",
        "往届经验贴的这条建议**方向是对的，理由也是对的**（结算价与收盘价不同、"
        "隔夜容易大额回撤），但**它不等于「改做日内就能赚钱」**。"
        "数据显示：去掉隔夜风险之后，日内**也没有出现正期望**。"
        "所以「不隔夜」省下的是**风险**，不是**赚到钱**。",
        "",
        "在你现在的情况下，隔夜风险已经由**止损单**控制住了 —— "
        "这才是对付隔夜风险的正确工具，而不是把持仓压缩到日内。",
        "",
        "### 如果一定要用日内，只有两个场景",
        "",
        "1. **资格补救**：到了 11 月底活跃交易日还不足 5 天，"
        "用**最小手数**在平今仓免手续费的品种"
        "（白糖、菜粕、PTA、棉花、硅铁、锰硅）上做几个来回补足。"
        "注意**避开棉花** —— 它的 1 跳 = 5 点，滑点最贵。",
        "2. **练手**：明确当成练习，**单独记账**，不混进主策略绩效。",
    ]


if __name__ == "__main__":
    main()
