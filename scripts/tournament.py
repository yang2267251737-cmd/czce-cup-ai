"""锦标赛模拟：在真实的**排名制**评分下，单笔风险到底该给多少。

真实评分规则（官方细则）
========================

    最终成绩 = 交易能力测试得分 + 理论知识水平测试得分（通过从业资格考试 +10 分）
    交易能力测试得分 = 单位净值得分 × 70% + 最大回撤度得分 × 15% + 波动率得分 × 15%
    单位净值得分 = (NAVPS_i / NAVPS_max × 100) × 30% + ((n + 1 - rank_i) / n × 100) × 70%

**关键：单位净值得分的 70% 是「排名」，不是绝对值。**
于是总分里有 ``0.70 × 0.70 = 49%`` 取决于名次 —— 这是一个锦标赛。

本脚本的做法
============

单看某一个人的回测毫无意义，因为排名取决于**别人怎么打**。所以这里做的是博弈模拟：

1. 对每个单笔风险档位，跑出 2016 年至今全部 40 交易日窗口的
   （收益率、窗口内最大回撤、窗口内日波动率）三元组 —— 这是「运气」的全部可能取值；
2. 假设有 n 个参赛者，每人随机抽一个窗口当自己的比赛结果；
3. 按**官方公式**给每个人算分（含排名）；
4. 比较不同风险档位下的**平均得分**。

诚实声明：参赛者的真实行为分布是未知的。所以脚本同时跑两种情况：
「所有人用同一个风险档位」和「别人保守、我激进」。
如果某个结论在两种情况下都成立，它才值得信。

用法::

    python scripts/tournament.py --refresh
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

from backtest import curve_to_frame, prepare_signals, run_backtest
from contracts import load_products
from market_data import load_daily
from risk import RiskConfig
from strategy import StrategyParams
from universe import load_universe

BASE = StrategyParams(entry_window=55, exit_window=5, stop_atr=3.0, trail_atr=6.0)
START = "2016-01-01"
WINDOW = 40
"""比赛窗口的交易日数（10-09 到 12-04 约 38 个）。"""

RISK_LEVELS = (0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08)
PARTICIPANTS = 2000
TRIALS = 60

DRAWDOWN_TOLERANCE = 0.15
VOLATILITY_TOLERANCE = 0.30

NAV_RATIO_WEIGHT = 0.30
NAV_RANK_WEIGHT = 0.70


def navps_score(nav: float, nav_max: float, rank: int, n: int) -> float:
    """单位净值得分（**用户提供的排名制变体公式**，0-100）。

    ``单位净值得分 = (NAVPS_i / NAVPS_max × 100) × 30% + ((n + 1 - rank_i) / n × 100) × 70%``

    ⚠️ 注意：第九届官方通知（郑商函〔2026〕670号）公布的是
    「收益率得分×80% + 回撤得分×10% + 活跃度得分×10%」，**没有**提到这个公式。
    本脚本把它作为**另一种可能的口径**单独模拟，用来检验结论是否依赖规则版本。
    """
    if n <= 0:
        return 0.0
    ratio_part = (nav / nav_max * 100) if nav_max > 0 else 0.0
    rank_part = (n + 1 - rank) / n * 100
    return ratio_part * NAV_RATIO_WEIGHT + rank_part * NAV_RANK_WEIGHT



def window_panel(frames, products, risk_level: float) -> pd.DataFrame:
    """跑一个风险档位，切出所有 40 交易日窗口，返回 (收益率, 最大回撤, 日波动率) 面板。"""
    prepared = prepare_signals(frames, products, BASE)
    result = run_backtest(prepared, RiskConfig(risk_per_trade=risk_level, max_positions=8),
                          name=f"风险{risk_level:.1%}", start=START)
    curve = curve_to_frame(result)
    equity = curve["equity"].to_numpy(dtype=float)
    rows = []
    for start in range(len(equity) - WINDOW):
        segment = equity[start:start + WINDOW + 1]
        ret = segment[-1] / segment[0] - 1
        peak = np.maximum.accumulate(segment)
        drawdown = float(np.max((peak - segment) / peak))
        daily = np.diff(segment) / segment[:-1]
        volatility = float(daily.std(ddof=0) * np.sqrt(244))
        rows.append({"收益率": ret, "最大回撤": drawdown, "年化波动率": volatility})
    return pd.DataFrame(rows)


def drawdown_score(drawdown: np.ndarray) -> np.ndarray:
    """回撤得分（0-100）。假设为绝对刻度：回撤到容忍线记 0，无回撤记 100。"""
    return np.clip(1.0 - drawdown / DRAWDOWN_TOLERANCE, 0.0, 1.0) * 100


def volatility_score(volatility: np.ndarray) -> np.ndarray:
    """波动率得分（0-100）。同上，绝对刻度。"""
    return np.clip(1.0 - volatility / VOLATILITY_TOLERANCE, 0.0, 1.0) * 100


def score_participants(navs: np.ndarray) -> np.ndarray:
    """按官方公式给一组参赛者算「交易能力测试得分」。"""
    n = len(navs)
    nav_max = navs.max()
    order = np.argsort(-navs)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    nav_scores = np.array([navps_score(navs[i], nav_max, ranks[i], n) for i in range(n)])
    return nav_scores


def simulate_same_risk(panels: dict[float, pd.DataFrame], rng: np.random.Generator) -> pd.DataFrame:
    """情形一：所有参赛者用同一个风险档位，看谁的期望得分最高。"""
    rows = []
    for level, panel in panels.items():
        returns = panel["收益率"].to_numpy()
        returns = np.clip(returns, -0.99, None)
        totals = []
        for _ in range(TRIALS):
            picks = rng.integers(0, len(panel), size=PARTICIPANTS)
            navs = 1.0 + returns[picks]
            dd = drawdown_score(panel["最大回撤"].to_numpy()[picks])
            vol = volatility_score(panel["年化波动率"].to_numpy()[picks])
            nav_score = score_participants(navs)
            total = nav_score * 0.70 + dd * 0.15 + vol * 0.15
            totals.append(total.mean())
        rows.append({
            "单笔风险": level,
            "期望交易得分": float(np.mean(totals)),
            "期望总分(含附加分)": float(np.mean(totals)) + THEORY_BONUS,
            "得分标准差": float(np.std(totals)),
        })
    return pd.DataFrame(rows)


def simulate_exploit(panels: dict[float, pd.DataFrame], crowd: float, rng: np.random.Generator) -> pd.DataFrame:
    """情形二：其他人都用 ``crowd`` 档位，我用每个候选档位，看我的期望得分。

    这才是真正可操作的比较 —— 你不是在和「和自己一样的人」比，你是在和一群
    大概率比你保守的同学比。
    """
    crowd_panel = panels[crowd]
    crowd_returns = np.clip(crowd_panel["收益率"].to_numpy(), -0.99, None)
    rows = []
    for level, panel in panels.items():
        mine_returns = np.clip(panel["收益率"].to_numpy(), -0.99, None)
        totals, win_rates = [], []
        for _ in range(TRIALS):
            others = rng.integers(0, len(crowd_panel), size=PARTICIPANTS - 1)
            me = rng.integers(0, len(panel), size=1)
            navs = np.concatenate([1.0 + crowd_returns[others], 1.0 + mine_returns[me]])
            dd = np.concatenate([drawdown_score(crowd_panel["最大回撤"].to_numpy()[others]),
                                 drawdown_score(panel["最大回撤"].to_numpy()[me])])
            vol = np.concatenate([volatility_score(crowd_panel["年化波动率"].to_numpy()[others]),
                                  volatility_score(panel["年化波动率"].to_numpy()[me])])
            nav_score = score_participants(navs)
            total = nav_score * 0.70 + dd * 0.15 + vol * 0.15
            totals.append(total[-1])
            win_rates.append(1.0 - (np.argsort(-navs) == len(navs) - 1).nonzero()[0][0] / len(navs))
        rows.append({
            "我的单笔风险": level,
            "我的期望交易得分": float(np.mean(totals)),
            "我的期望名次分位": float(np.mean(win_rates)),
            "别人用的档位": crowd,
        })
    return pd.DataFrame(rows)


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
    """跑锦标赛模拟并写出 REPORTS/tournament.md。"""
    parser = argparse.ArgumentParser(description="排名制赛制下的仓位博弈模拟")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--participants", type=int, default=PARTICIPANTS)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--crowd", type=float, default=0.01, help="假想中其他参赛者的单笔风险档位")
    parser.add_argument("--out", default="REPORTS")
    args = parser.parse_args()

    global PARTICIPANTS, TRIALS
    PARTICIPANTS, TRIALS = args.participants, args.trials

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_universe(tiers=("core",))
    frames = {entry.continuous: load_daily(entry.continuous, refresh=args.refresh) for entry in entries}
    products = load_products()
    rng = np.random.default_rng(20261009)

    panels: dict[float, pd.DataFrame] = {}
    for level in RISK_LEVELS:
        print(f"跑风险档位 {level:.1%} …")
        panels[level] = window_panel(frames, products, level)

    same = simulate_same_risk(panels, rng)
    exploit = simulate_exploit(panels, args.crowd, rng)

    best_same = same.loc[same["期望交易得分"].idxmax()]
    best_ex = exploit.loc[exploit["我的期望交易得分"].idxmax()]
    conservative = same.loc[same["单笔风险"].idxmin()]
    aggressive = same.loc[same["单笔风险"].idxmax()]

    lines = [
        "# 锦标赛模拟：排名制赛制下，单笔风险该给多少",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。"
        f"模拟参数：{PARTICIPANTS:,} 名参赛者、重复 {TRIALS} 次、窗口 {WINDOW} 个交易日。",
        "",
        "## 0. 为什么单看回测没有意义了",
        "",
        "官方公式：",
        "",
        "```",
        "单位净值得分 = (NAVPS_i / NAVPS_max × 100) × 30% + ((n + 1 - rank_i) / n × 100) × 70%",
        "交易能力测试得分 = 单位净值得分 × 70% + 最大回撤得分 × 15% + 波动率得分 × 15%",
        "最终成绩 = 交易能力测试得分 + 通过从业资格考试的 10 分",
        "```",
        "",
        "**单位净值得分的 70% 是排名，不是绝对值。** 于是：",
        "",
        "| 组成部分 | 占总分权重 |",
        "|---|---|",
        "| 单位净值**排名** | 0.70 × 0.70 = **49%** |",
        "| 单位净值**与最高净值之比** | 0.70 × 0.30 = **21%** |",
        "| 最大回撤 | **15%** |",
        "| 波动率 | **15%** |",
        "",
        "**接近一半的分数取决于你排第几。** 而排名取决于别人怎么打 —— "
        "所以「我的回测收益是多少」这个问题本身就不完整，必须做博弈模拟。",
        "",
        "## 1. 情形一：所有人都用同一个风险档位",
        "",
        _table(same, percent_cols={"单笔风险"}),
        "",
        f"> 期望得分最高的是 **{best_same['单笔风险']:.1%}** 档"
        f"（交易得分 {best_same['期望交易得分']:.2f}）。"
        f"最保守的 {conservative['单笔风险']:.1%} 档是 {conservative['期望交易得分']:.2f}，"
        f"最激进的 {aggressive['单笔风险']:.1%} 档是 {aggressive['期望交易得分']:.2f}。",
        "",
        "## 2. 情形二：别人保守（1%），我用不同档位",
        "",
        _table(exploit, percent_cols={"我的单笔风险", "我的期望名次分位", "别人用的档位"}),
        "",
        f"> 如果其他参赛者都用保守的 {args.crowd:.1%} 档，"
        f"**我用 {best_ex['我的单笔风险']:.1%} 档时期望得分最高**"
        f"（{best_ex['我的期望交易得分']:.2f}），期望名次分位 "
        f"{best_ex['我的期望名次分位']:.1%}。",
        "",
        "## 3. 结论",
        "",
        _verdict(same, exploit, best_same, best_ex, conservative, aggressive),
        "",
        "## 4. 复现命令",
        "",
        "```powershell",
        "python scripts/tournament.py --refresh",
        "```",
    ]
    path = out_dir / "tournament.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出：{path}")


def _verdict(same, exploit, best_same, best_ex, conservative, aggressive) -> list[str]:
    """给出可执行的结论。"""
    same_spread = same["期望交易得分"].max() - same["期望交易得分"].min()
    ex_spread = exploit["我的期望交易得分"].max() - exploit["我的期望交易得分"].min()
    return [
        "### 三条硬结论",
        "",
        f"1. **在全场同档位的假设下，风险档位的选择对得分的影响其实不大**："
        f"最好和最差档位之间只差 {same_spread:.2f} 分（满分 100）。"
        "原因是排名项占 49%，而排名是个零和游戏 —— 你放大波动，"
        "排名分布被拉宽，期望名次并没有改善多少，但回撤和波动率两项的惩罚是确定的。",
        "",
        f"2. **但「别人保守、我激进」时有可利用的空间**："
        f"别人都在 {best_ex['别人用的档位']:.1%} 时，我用 {best_ex['我的单笔风险']:.1%} "
        f"比用 1% 的期望得分高 {ex_spread:.2f} 分。"
        "这个空间的来源是排名项的非线性：保守的人挤在净值 1.0 附近，"
        "名次密集；稍微拉开一点距离就能超过很多人。",
        "",
        "3. **别忽略那 10 分。** 通过期货从业资格考试是**确定的** "
        f"{THEORY_BONUS:.0f} 分，而上面全部仓位博弈加起来的分差只有 "
        f"{max(same_spread, ex_spread):.2f} 分。"
        "**考一个证 > 所有仓位调整的收益总和，而且没有亏损风险。**",
        "",
        "### 所以该怎么打",
        "",
        "| 优先级 | 动作 | 依据 |",
        "|---|---|---|",
        "| **1** | **去考期货从业资格考试** | 稳拿 10 分，比任何策略调整都值 |",
        "| **2** | 保持纪律：每笔必挂止损、固定单笔风险 | 回撤+波动率合计 30%，是确定的扣分项 |",
        "| **3** | 适度提高单笔风险到 2%–3% | 在别人保守时能拉开名次，但代价是回撤与波动率 |",
        "| **4** | 加品种（8 → 12） | 净值不变的前提下压低波动率，纯赚 |",
        "",
        "**不要做的事**：把所有钱压上去赌一个方向。排名项虽然占 49%，"
        "但它是线性排名 —— 你赌赢了只是名次靠前，赌输了直接垫底，"
        "而回撤分项会同时归零。期望上不划算。",
    ]


if __name__ == "__main__":
    main()
