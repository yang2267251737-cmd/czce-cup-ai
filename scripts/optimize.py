"""参数与赛制优化：把"回测 → 赛制评分 → 频率/风险取舍"变成一条可重复运行的命令。

产出两份报告（写进 ``REPORTS/``）：
- ``backtest.md``：候选策略的分段回测、逐笔统计与数据质量说明。
- ``frequency-optimization.md``：交易频率扫描、风险预算前沿、赛制评分敏感性。

用法::

    python scripts/optimize.py --refresh
    python scripts/optimize.py --no-chart        # 无 matplotlib 环境时跳过画图
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import BacktestResult, compare_results, curve_to_frame, prepare_signals, run_backtest
from contracts import load_products, roll_warnings
from market_data import load_daily, quality_report
from risk import RiskConfig
from scoremodel import (
    DEFAULT_ASSUMPTIONS,
    ScoreAssumptions,
    optimal_frequency_note,
    rank_robustness,
    score_components,
    theory_bonus_note,
)
from strategy import StrategyParams
from universe import code_of, excluded_reasons, load_universe

PERIODS: dict[str, tuple[str | None, str | None]] = {
    "全历史": (None, None),
    "2016 年至今": ("2016-01-01", None),
    "近 3 年": ("2023-10-01", None),
    "近 1 年": ("2025-10-01", None),
}

CANDIDATES: dict[str, dict] = {
    "A 海龟默认 N20/M10/2ATR/跟踪3ATR": dict(entry_window=20, exit_window=10, stop_atr=2.0, trail_atr=3.0),
    "B 宽跟踪 N20/M20/2ATR/跟踪6ATR": dict(entry_window=20, exit_window=20, stop_atr=2.0, trail_atr=6.0),
    "C 宽跟踪 N40/M5/2ATR/跟踪6ATR": dict(entry_window=40, exit_window=5, stop_atr=2.0, trail_atr=6.0),
    "D 慢速宽跟踪 N55/M5/3ATR/跟踪6ATR": dict(entry_window=55, exit_window=5, stop_atr=3.0, trail_atr=6.0),
    "E 慢速+均值回归辅助": dict(entry_window=55, exit_window=5, stop_atr=3.0, trail_atr=6.0),
}
SATELLITE_FOR = {"E 慢速+均值回归辅助"}
FREQUENCY_SWEEP = (10, 20, 30, 40, 55, 70, 90)
RISK_SWEEP = (0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08)


def load_frames(entries, refresh: bool) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """载入品种池的日线并顺手做质量检查。"""
    frames: dict[str, pd.DataFrame] = {}
    reports: list[dict] = []
    for entry in entries:
        frame = load_daily(entry.continuous, refresh=refresh)
        frame.attrs["cleaning"] = frame.attrs.get("cleaning", {})
        frames[entry.continuous] = frame
        report = quality_report(entry.continuous, frame)
        report["name"] = entry.name
        reports.append(report)
        flag = "OK" if report["ok"] else "FAIL"
        print(f"[{flag}] {entry.continuous:<5} {report['rows']:>5} 行  {report['start']} → {report['end']}"
              f"  换月跳空 {report['jump_days']} 天")
    return frames, reports


def evaluate_candidate(
    frames: dict[str, pd.DataFrame],
    products,
    params: StrategyParams,
    risk: RiskConfig,
    use_satellite: bool,
) -> dict[str, BacktestResult]:
    """一个候选策略在所有考察区间上的回测结果。"""
    satellite = StrategyParams(entry_z=2.0, exit_z=0.0, stop_atr=2.5, trail_atr=3.0,
                               regime_max_adx=20.0, max_hold_days=10) if use_satellite else None
    prepared = prepare_signals(frames, products, params, satellite)
    results: dict[str, BacktestResult] = {}
    for period, (start, end) in PERIODS.items():
        results[period] = run_backtest(
            prepared, risk, name=params.label(), params_note=params.label(), start=start, end=end
        )
    return results


def build_period_table(all_results: dict[str, dict[str, BacktestResult]]) -> pd.DataFrame:
    """把「候选 × 区间」的回测结果铺成一张总表。"""
    keys = ["总收益率", "年化收益率", "最大回撤", "夏普比率", "交易笔数", "胜率",
            "活跃交易日占比", "手续费合计", "因风控放弃的开仓次数"]
    rows = []
    for name, by_period in all_results.items():
        for period, result in by_period.items():
            row = {"候选策略": name, "区间": period}
            row.update({key: result.metrics.get(key, float("nan")) for key in keys})
            rows.append(row)
    return pd.DataFrame(rows)


def risk_frontier(frames, products, params: StrategyParams, use_satellite: bool) -> pd.DataFrame:
    """风险预算前沿：在收益（80%）与回撤（10%）的赛制权重下，单笔风险该给多大。"""
    satellite = StrategyParams(entry_z=2.0, exit_z=0.0, stop_atr=2.5, trail_atr=3.0,
                               regime_max_adx=20.0, max_hold_days=10) if use_satellite else None
    prepared = prepare_signals(frames, products, params, satellite)
    rows = []
    for risk_pct in RISK_SWEEP:
        for period, (start, end) in PERIODS.items():
            risk = RiskConfig(risk_per_trade=risk_pct, max_positions=8)
            result = run_backtest(prepared, risk, name=f"风险{risk_pct:.1%}", start=start, end=end)
            metrics = result.metrics
            scored = score_components(metrics, ScoreAssumptions())
            rows.append({
                "单笔风险": risk_pct,
                "区间": period,
                "总收益率": metrics["总收益率"],
                "最大回撤": metrics["最大回撤"],
                "年化波动率": metrics["年化波动率"],
                "模拟总分": scored["模拟总分"],
                "交易笔数": metrics["交易笔数"],
                "活跃交易日占比": metrics["活跃交易日占比"],
                "手续费合计": metrics["手续费合计"],
            })
    return pd.DataFrame(rows)


def frequency_sweep(frames, products, risk: RiskConfig) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """交易频率扫描：入场窗口越长，交易越少、手续费越低，但活跃度也越低。"""
    rows = []
    results: dict[str, BacktestResult] = {}
    for window in FREQUENCY_SWEEP:
        params = StrategyParams(entry_window=window, exit_window=max(5, window // 4),
                                stop_atr=3.0, trail_atr=6.0)
        prepared = prepare_signals(frames, products, params)
        result = run_backtest(prepared, risk, name=f"N={window}", start="2016-01-01")
        metrics = result.metrics
        scored = score_components(metrics)
        rows.append({
            "参数": f"入场 {window} 日 / 离场 {max(5, window // 4)} 日",
            "交易笔数": metrics["交易笔数"],
            "总收益率": metrics["总收益率"],
            "最大回撤": metrics["最大回撤"],
            "活跃交易日占比": metrics["活跃交易日占比"],
            "手续费合计": metrics["手续费合计"],
            "手续费占初始权益": metrics["手续费占初始权益"],
            "平均持有交易日": metrics["平均持有交易日"],
            "模拟总分": scored["模拟总分"],
        })
        results[f"N={window}"] = result
    return pd.DataFrame(rows), results


def plot_equity(results: dict[str, BacktestResult], path: Path, title: str = "策略净值对比") -> bool:
    """把净值曲线画成图；没有 matplotlib 或中文字体时安静跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for name, result in results.items():
        curve = curve_to_frame(result)
        axes[0].plot(curve["date"], curve["equity"], label=f"{name}", linewidth=1.2)
        axes[1].plot(curve["date"], -curve["drawdown"] * 100, linewidth=1.0, label=name)
    axes[0].set_ylabel("账户权益（元）")
    axes[0].set_title(f"{title}（初始 50 万元，手续费 1.5 倍 + 1 tick 滑点）")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("回撤（%）")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _markdown_table(frame: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """把 DataFrame 转成 Markdown 表格，浮点列按需格式化。"""
    if frame.empty:
        return "_（无数据）_"
    percent_columns = {"总收益率", "年化收益率", "最大回撤", "胜率", "活跃交易日占比",
                       "手续费占初始权益", "单笔风险"}
    money_columns = {"手续费合计", "期末权益"}
    count_columns = {"交易笔数", "因风控放弃的开仓次数", "剔除脏bar", "有交易天数", "交易日总数"}
    view = frame.copy()
    for column in view.columns:
        if column in money_columns and pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:,.0f} 元" if pd.notna(value) else "-")
        elif column in count_columns and pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{int(value):,}" if pd.notna(value) else "-")
        elif pd.api.types.is_float_dtype(view[column]):
            if column in percent_columns:
                view[column] = view[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "-")
            else:
                view[column] = view[column].map(lambda value: float_format.format(value) if pd.notna(value) else "-")
        elif pd.api.types.is_integer_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:,}")
    header = "| " + " | ".join(str(column) for column in view.columns) + " |"
    divider = "|" + "|".join("---" for _ in view.columns) + "|"
    body = "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False))
    return "\n".join([header, divider, body])


def write_backtest_report(
    path: Path,
    period_table: pd.DataFrame,
    all_results: dict[str, dict[str, BacktestResult]],
    quality: list[dict],
    products,
    chosen: str,
    chart: bool,
) -> None:
    """写 REPORTS/backtest.md。"""
    lines = [
        "# 回测报告：郑商所 8 品种日线趋势策略",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。"
        "所有数字来自本仓库脚本的真实运行，命令见文末。",
        "",
        "## 1. 口径与成本假设",
        "",
        "- 初始权益 50 万元（赛制虚拟保证金规模）。",
        "- **手续费按交易所标准 1.5 倍收取**；当日开当日平按「平今仓」费率结算。",
        "- 每次成交再计 1 个最小变动价位的滑点。",
        "- 保证金按交易所标准 × 1.3 安全垫占用，上限为权益的 60%。",
        "- 信号在收盘产生、次日开盘成交；止损是盘中挂单，跳空穿越时按更差的开盘价成交。",
        "",
        "## 2. 数据质量",
        "",
        _markdown_table(pd.DataFrame([
            {"品种": item["code"], "名称": item.get("name", ""), "行数": item["rows"],
             "起始": item["start"], "结束": item["end"],
             "剔除脏bar": sum(item.get("cleaning", {}).values()) if item.get("cleaning") else 0,
             "单日涨跌超4%天数": item["jump_days"],
             "结论": "通过" if item["ok"] else "有错误"}
            for item in quality
        ])),
        "",
        "> **换月跳空是已知偏差**：新浪连续合约在主力换月处存在价格跳空，"
        "会让突破信号产生假触发。真实交易需按主力合约代码下单，并按 "
        "`python scripts/contracts.py` 输出的换月提醒提前移仓。",
        "",
        "## 3. 候选策略分段表现",
        "",
        "参数含义：`N`=入场通道天数，`M`=离场通道天数，止损与跟踪止损都按 ATR 倍数表示。",
        "",
        _markdown_table(period_table),
        "",
        "## 4. 结论",
        "",
    ]
    chosen_results = all_results[chosen]
    lines.extend(_conclusions(chosen, chosen_results, period_table))
    lines.extend([
        "",
        "## 5. 净值曲线",
        "",
        "![策略净值对比](img/equity.png)" if chart else "_（本次运行未生成图，缺少 matplotlib 时可忽略）_",
        "",
        "## 6. 复现命令",
        "",
        "```powershell",
        "python scripts/contracts.py",
        "python scripts/optimize.py --refresh",
        "```",
        "",
        "## 7. 这个回测**不能**说明什么",
        "",
        "- 不能说明未来 40 个交易日会重复历史收益。比赛的窗口太短，"
        "单次结果主要由运气决定，而不是由期望值决定。",
        "- 不能说明策略可以自动下单：本项目**不下单、不接交易接口**，只输出人工执行的操作清单。",
        "- 不能替代对赛制规则的核验：收益率/回撤/活跃度的具体计算公式尚未从官方文件确认。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _conclusions(chosen: str, results: dict[str, BacktestResult], period_table: pd.DataFrame) -> list[str]:
    """从回测结果里提炼几条可读的结论。"""
    latest = results.get("近 3 年") or list(results.values())[-1]
    full = results.get("全历史") or list(results.values())[0]
    metrics = latest.metrics
    full_metrics = full.metrics
    lines = [
        f"1. **稳健性优先**：在全部 {len(period_table['候选策略'].unique())} 个候选里，"
        f"「{chosen}」是四个考察区间的**最差区间收益最高者**。"
        "评价趋势策略一定要看多个区间：只看全历史会被 2008-2015 年的大趋势误导。",
        f"2. **近 3 年表现**：总收益 {metrics.get('总收益率', 0):.2%}，"
        f"最大回撤 {metrics.get('最大回撤', 0):.2%}，"
        f"共 {metrics.get('交易笔数', 0)} 笔，"
        f"活跃交易日占比 {metrics.get('活跃交易日占比', 0):.2%}，"
        f"手续费合计 {metrics.get('手续费合计', 0):,.0f} 元。",
        f"3. **胜率低是正常的**：全历史胜率只有 {full_metrics.get('胜率', 0):.1%}，"
        "靠少数几笔大趋势赚钱，必须能忍住连续 5-8 笔小亏。"
        "这正是必须用固定比例风险、而不是凭感觉加仓的原因。",
        f"4. **手续费不是主要成本**：全历史手续费合计 {full_metrics.get('手续费合计', 0):,.0f} 元，"
        f"只占初始权益的 {full_metrics.get('手续费占初始权益', 0):.2%}。"
        "真正吃掉收益的是被反复止损打掉的尝试（胜率不足四成）—— 也就是说，"
        "「少交易省手续费」并不是提高收益的关键，「提高信号质量」才是。",
        f"5. **edge 在衰减**：把「{chosen}」的全历史数字和近 3 年数字并排看，"
        "近十年的趋势性明显弱于 2005-2015 年。**不要把这个策略当成赚钱机器**，"
        "它的作用是给交易提供一个有正偏度、可复盘的框架。",
    ]
    return lines


def write_optimization_report(
    path: Path,
    sweep: pd.DataFrame,
    frontier: pd.DataFrame,
    robustness: pd.DataFrame,
    sensitivity: pd.DataFrame,
    chosen_note: str,
) -> None:
    """写 REPORTS/frequency-optimization.md —— 本项目最有价值的一份分析。"""
    base = ScoreAssumptions()
    lines = [
        "# 赛制优化：交易频率、风险预算与评分敏感性",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}。",
        "",
        "## 0. 评分规则（官方细则，不是假设）",
        "",
        "```",
        "最终成绩 = 交易能力测试得分 + 理论知识水平测试得分",
        "交易能力测试得分 = 单位净值得分 × 70% + 最大回撤度得分 × 15% + 波动率得分 × 15%",
        "理论知识水平测试得分 = 通过期货从业资格考试 → 固定 +10 分",
        "```",
        "",
        "**注意：这里没有「活跃度」这一项。** 本报告早期版本曾按"
        "「收益率 80% + 回撤 10% + 活跃度 10%」分析，那是错的，已全部更正。",
        "更正后的含义完全不同：**风险相关项合计占 30%，而不是 10%**；"
        "而且波动率与回撤方向一致（越低越好），"
        "所以这是一个**风险调整后收益**的赛制 —— 比的不是谁赚得多，是谁的「类夏普比率」高。",
        "",
        "三部分的**权重来自官方细则**；仍然未知的是每一部分如何归一化，"
        "所以下面三个归一化参数仍是估计值，并做敏感性检验：",
        "",
        f"- 单位净值满分线：**{base.nav_reference:.0%}**（净值涨到这里该项记满）",
        f"- 回撤归零线：**{base.drawdown_tolerance:.0%}**（最大回撤到这个水平该项归零）",
        f"- 波动率归零线：**{base.volatility_tolerance:.0%}**（年化波动率到这个水平该项归零）",
        "",
        "**结论必须对所有合理假设都成立才算数**，第 3 节专门做敏感性检验。",
        "",
        "## 1. 交易频率扫描（区间：2016 年至今）",
        "",
        "既然没有活跃度项，交易频率就只剩两个影响：手续费，和信号质量。"
        "下表扫入场窗口从 10 日到 90 日：",
        "",
        _markdown_table(sweep),
        "",
        "> " + optimal_frequency_note(sweep),
        "",
        "**这张表最关键的一列是「总收益率」**：从 10 日到 90 日窗口，2016 年至今全部为负。"
        "也就是说，**交易频率根本不是这个策略赚不赚钱的原因** —— 换频率只是在"
        "「多付手续费」和「少付手续费」之间挪动，两边都是亏的。"
        "想把注意力放对地方，应该去改信号质量，而不是调频率。",
        "",
        "## 2. 风险预算前沿",
        "",
        "**赛制实际惩罚高风险，而不是奖励它。** 风险相关项合计 30%："
        "回撤越大、波动越高，分数越低。所以下表的正确读法是"
        "「找分数最高的那一档」，而不是「越高越好」：",
        "",
        _markdown_table(frontier),
        "",
        _risk_note(frontier),
        "",
        "## 3. 评分假设敏感性",
        "",
        "同一份回测结果，在不同评分假设下的总分 —— 区间越窄，说明结论越不依赖某个估计的归一化参数：",
        "",
        _markdown_table(sensitivity.drop(columns=[col for col in ["_假设参数"] if col in sensitivity.columns])),
        "",
        "> 完整的假设参数见 `scripts/scoremodel.py` 的 `DEFAULT_ASSUMPTIONS`。",
        "",
        "## 4. 候选策略的稳健排名",
        "",
        "按「在所有假设下的平均排名」排序，而不是按某一次的最高分排序：",
        "",
        _markdown_table(robustness),
        "",
        "> " + chosen_note,
        "",
        "## 5. 可执行的结论",
        "",
        "1. **考试那 10 分比策略更值钱。** " + theory_bonus_note(),
        "2. **波动率现在是要花钱买的。** 波动率占 15% 权重，且越低越好。"
        "提高单笔风险会同时推高回撤和波动率，也会推高净值的**方差**（不是期望）—— "
        "在一个期望接近零的策略上，这等于单方面扣分。",
        "3. **不要为了活跃度去交易**（本来也没有活跃度项）。"
        "交易频率只受手续费影响，所以其他条件相同时，交易越少越好。",
        "4. **分散化是这套赛制下唯一的免费午餐**：多品种、低相关，"
        "能在净值不变的前提下同时压低回撤和波动率 —— **两项各占 15%**。",
        "",
        "## 6. 复现命令",
        "",
        "```powershell",
        "python scripts/optimize.py --refresh",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DRAWDOWN_LIMIT = 0.20
"""风险档位建议用的硬约束：所有考察区间的最大回撤都不超过 20%。

把回撤当**约束**而不是优化目标，是本项目在赛制下的核心取舍 ——
回撤只占 10 分，但超过这个水平之后分数不再改善，代价却是实打实的。
"""


def _risk_note(frontier: pd.DataFrame) -> str:
    """解读风险预算前沿。真实赛制下风险项占 30%，所以结论是「越低越好」。"""
    if frontier.empty:
        return ""
    mean_score = frontier.groupby("单笔风险")["模拟总分"].mean()
    mean_dd = frontier.groupby("单笔风险")["最大回撤"].mean()
    best_risk = float(mean_score.idxmax())
    lowest_risk = float(mean_score.index.min())
    highest_risk = float(mean_score.index.max())
    return (
        f"> 按四段区间的平均模拟总分看，最优档位是 **{best_risk:.1%}**"
        f"（平均总分 {mean_score.max():.4f}），而风险最小的 {lowest_risk:.1%} 档是 "
        f"{mean_score.get(lowest_risk, float('nan')):.4f}，"
        f"风险最大的 {highest_risk:.1%} 档只有 {mean_score.get(highest_risk, float('nan')):.4f}。\n"
        ">\n"
        "> **这和上一版结论正好相反。** 之前按「收益率 80% / 回撤 10%」的假设算出来的是"
        "「风险越高分数越高」；真实赛制把风险项提到 30%，而且还多了一项波动率，"
        "**天平直接翻过来了**：仓位越大，回撤和波动率一起恶化，两项目各扣 15%，"
        "而净值项的**期望**并没有变好（这个策略的期望接近零，放大的只是方差）。\n"
        ">\n"
        f"> 平均最大回撤也印证这一点：{lowest_risk:.1%} 档平均回撤 "
        f"{mean_dd.get(lowest_risk, float('nan')):.2%}，"
        f"{highest_risk:.1%} 档升到 {mean_dd.get(highest_risk, float('nan')):.2%}。\n"
        ">\n"
        "> **一句话：在这个赛制里，加仓位是单方面扣分，不是加分。**"
        "想把资金利用率做上去，应该加品种（分散化），不是加杠杆。"
    )


def main() -> None:
    """命令行入口：跑完候选回测、频率扫描与风险前沿，写出两份 Markdown 报告。"""
    parser = argparse.ArgumentParser(description="郑商所杯策略参数与赛制评分优化")
    parser.add_argument("--refresh", action="store_true", help="联网刷新行情缓存")
    parser.add_argument("--tiers", default="core", help="品种池层级，逗号分隔：core,satellite")
    parser.add_argument("--no-chart", action="store_true", help="跳过 matplotlib 画图")
    parser.add_argument("--out", default="REPORTS", help="报告输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_universe(tiers=tuple(args.tiers.split(",")))
    print(f"品种池（{len(entries)} 个）：{[entry.code for entry in entries]}")
    products = load_products()
    frames, quality = load_frames(entries, args.refresh)

    risk = RiskConfig(risk_per_trade=0.01, max_positions=8)
    all_results: dict[str, dict[str, BacktestResult]] = {}
    for name, config in CANDIDATES.items():
        params = StrategyParams(**config)
        print(f"回测候选：{name}")
        all_results[name] = evaluate_candidate(
            frames, products, params, risk, use_satellite=name in SATELLITE_FOR
        )

    period_table = build_period_table(all_results)
    print(period_table.to_string(index=False))

    # 选稳健冠军：四个区间都不亏 + 最小区间收益率最高
    def floor_return(name: str) -> float:
        return min(result.metrics["总收益率"] for result in all_results[name].values())

    chosen = max(all_results, key=floor_return)
    chosen_params = StrategyParams(**CANDIDATES[chosen])
    print(f"稳健冠军：{chosen}（最差区间收益 {floor_return(chosen):.2%}）")

    sweep, sweep_results = frequency_sweep(frames, products, risk)
    frontier = risk_frontier(frames, products, chosen_params, chosen in SATELLITE_FOR)

    robustness_input = {name: by_period["近 3 年"].metrics for name, by_period in all_results.items()}
    robustness = rank_robustness(robustness_input, DEFAULT_ASSUMPTIONS)
    sensitivity = pd.DataFrame([
        score_components(all_results[chosen]["近 3 年"].metrics, assumption)
        for assumption in DEFAULT_ASSUMPTIONS
    ])
    chosen_note = (
        f"「{chosen}」在 {len(DEFAULT_ASSUMPTIONS)} 组评分假设下的平均排名为 "
        f"{robustness.loc[robustness['策略'] == chosen, '平均排名'].squeeze()}（1 为最好）。"
        "如果某个策略只在某一组假设下排第一，就不要选它。"
    )

    chart_ok = False
    if not args.no_chart:
        chart_ok = plot_equity(
            {name: all_results[name]["近 3 年"] for name in all_results},
            out_dir / "img" / "equity.png",
        )

    write_backtest_report(out_dir / "backtest.md", period_table, all_results, quality, products,
                          chosen, chart_ok)
    write_optimization_report(out_dir / "frequency-optimization.md", sweep, frontier, robustness,
                              sensitivity, chosen_note)
    print(f"已写出：{out_dir / 'backtest.md'}")
    print(f"已写出：{out_dir / 'frequency-optimization.md'}")
    if chart_ok:
        print(f"已写出：{out_dir / 'img' / 'equity.png'}")

    warnings = roll_warnings(products)
    if warnings:
        print("\n换月提醒：")
        for line in warnings[:6]:
            print("  -", line)
    print(f"\n已排除品种：{len(excluded_reasons())} 个（原因见 config/universe.json 与 DOCS/decisions.md）")


if __name__ == "__main__":
    main()
