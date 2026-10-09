"""每日交易计划：每个交易日收盘后跑一次，产出次日**可以直接照着挂单**的操作清单。

这是整个项目的交付物本体。它只做决策辅助，**不下单、不接交易接口**：
所有价格都是供人工在大赛专用交易软件里输入的条件单参考价。

一份计划的生成流程::

    日线（含最新已收盘 bar）
      → 指标特征
      → 唐奇安突破状态机（单品种该拿什么方向、止损在哪）
      → 组合层：按当前权益、单笔风险、回撤状态算手数
      → 次一交易日的挂单清单 + 时刻表 + 复盘模板

核心命令::

    python scripts/daily_plan.py --refresh          # 刷新行情并生成次日计划
    python scripts/daily_plan.py --asof 2026-10-12  # 指定目标交易日（复盘用）
    python scripts/daily_plan.py --derive           # 用回测口径的"系统组合"代替手工账簿

**免责声明**：本清单是比赛用虚拟账户的决策辅助材料，不构成投资建议；
所有数字基于历史数据的统计推断，不保证未来结果。执行前请自行核对赛制规则。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import prepare_signals, run_backtest
from contracts import load_products, roll_warnings
from indicators import compute_features, describe_snapshot
from market_data import last_completed_bar, load_daily, quality_report, stale_warning
from risk import RiskConfig, portfolio_guard, size_position
from strategy import StrategyParams, breakout_positions, next_bar_plan
from universe import load_universe

STATE_PATH = ROOT / "state" / "portfolio.json"
REPORT_DIR = ROOT / "REPORTS" / "daily"

DEFAULT_STRATEGY = StrategyParams(entry_window=55, exit_window=5, stop_atr=3.0, trail_atr=6.0)
"""默认参数取优化脚本选出的稳健档（慢速宽跟踪）：交易频率低、抗噪、回撤小。"""


@dataclass
class Book:
    """账户账簿：当前权益、历史高点与在手持仓。"""

    equity: float = 500_000.0
    peak_equity: float = 500_000.0
    positions: list[dict] = field(default_factory=list)
    last_updated: str = ""
    notes: str = ""

    @property
    def drawdown(self) -> float:
        """当前回撤（相对历史高点）。"""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def open_codes(self) -> set[str]:
        """已持仓的品种代码集合。"""
        return {position["code"] for position in self.positions}


def load_book(path: Path = STATE_PATH, create_if_missing: bool = True) -> Book:
    """读取账户账簿；不存在时创建一个空仓账簿（初始权益 50 万）。"""
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("_note", None)
        return Book(**payload)
    book = Book(
        equity=500_000.0,
        peak_equity=500_000.0,
        positions=[],
        last_updated=date.today().isoformat(),
        notes="空仓起步。每次实际成交后请手工更新本文件，daily_plan 会据此计算手数与风控状态。",
    )
    if create_if_missing:
        save_book(book, path)
    return book


def save_book(book: Book, path: Path = STATE_PATH) -> None:
    """把账簿写回磁盘，附带一段说明。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_note": "账户账簿：每次实际成交后手工更新。scripts/daily_plan.py 会读取它。", **asdict(book)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def book_from_backtest(result) -> Book:
    """用回测结束时的持仓构造"系统跟踪组合"，用于对照手工账簿。"""
    curve = result.equity_curve
    equity = float(curve["equity"].iloc[-1]) if len(curve) else result.initial_equity
    return Book(
        equity=round(equity, 2),
        peak_equity=round(float(curve["equity"].cummax().iloc[-1]), 2) if len(curve) else equity,
        positions=[
            {
                "code": item["code"],
                "direction": item["direction"],
                "lots": item["lots"],
                "entry_price": item["entry_price"],
                "entry_date": item["entry_date"],
                "stop": item["stop"],
            }
            for item in result.final_positions
        ],
        last_updated=date.today().isoformat(),
        notes="由回测口径推导的跟踪组合，仅供对照，不代表真实成交。",
    )


def estimate_next_trading_day(frames: dict[str, pd.DataFrame], now: datetime | None = None) -> tuple[date, date]:
    """推断「下一个交易日」与「最后一根已收盘日线」的日期。

    规则：目标交易日 = 最后一根**已收盘**日线的**下一个**交易日。
    收盘后运行时会自然得到次日；盘中运行会得到当天（此时清单已过半程，会额外告警）。
    周末会被跳过，但**法定节假日无法从行情推断**，需要用户自己核对（函数返回的告警会说明）。
    """
    now = now or datetime.now()
    latest: date | None = None
    for frame in frames.values():
        bar_date = pd.to_datetime(last_completed_bar(frame, now)["date"]).date()
        latest = bar_date if latest is None else max(latest, bar_date)
    if latest is None:
        raise RuntimeError("没有任何可用行情")
    candidate = latest + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate, latest


def execution_window(target: date, has_night: bool, last_bar_date: date) -> str:
    """给出该品种下一次可以成交的具体时间窗。

    夜盘属于**下一个交易日**，所以 target 的夜盘发生在「上一根已收盘日线」那天晚上。
    """
    if has_night:
        return (f"{last_bar_date.isoformat()}（{_weekday_cn(last_bar_date)}）21:00 夜盘开盘 —— "
                f"夜盘属于 {target.isoformat()} 这个交易日")
    return f"{target.isoformat()}（{_weekday_cn(target)}）09:00 日盘开盘"


def _weekday_cn(day: date) -> str:
    return "周" + "一二三四五六日"[day.weekday()]


def build_plan(
    entries,
    products,
    frames: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame],
    states: dict[str, pd.DataFrame],
    book: Book,
    risk: RiskConfig,
    params: StrategyParams,
    target_day: date,
    last_bar_date: date,
) -> dict:
    """把行情、策略状态与账户账簿合成一份完整的次日操作方案。"""
    allowed, guard_note, risk_multiplier = portfolio_guard(
        book.equity, risk, len(book.positions), _used_margin(book, products), book.drawdown
    )
    remaining_margin = book.equity * risk.max_margin_usage - _used_margin(book, products)

    new_orders: list[dict] = []
    manage: list[dict] = []
    blocked: list[dict] = []
    market: list[dict] = []

    for entry in entries:
        code = entry.continuous
        product_code = entry.code
        product = products[product_code]
        plan = next_bar_plan(features[code], params, states[code], code)
        snapshot = describe_snapshot(features[code])
        last_bar = features[code].dropna(subset=["atr"]).iloc[-1]
        price = float(last_bar["close"])
        atr_now = float(last_bar["atr"])

        market.append({
            "代码": product_code, "名称": entry.name, "收盘": price,
            "ATR": atr_now, "ATR%": atr_now / price,
            "ADX": snapshot["adx"], "状态": snapshot["regime"],
            "波动分位": snapshot["vol_pctile"],
            "距突破": (plan.entry_trigger - price) / price if plan.direction == 0 and np.isfinite(plan.entry_trigger) else np.nan,
            "当前方向": int(states[code]["target_state"].iloc[-1]),
        })

        held = next((position for position in book.positions if position["code"] == product_code), None)
        if held is not None:
            stop = float(plan.stop) if np.isfinite(plan.stop) else held.get("stop")
            manage.append({
                "代码": product_code, "名称": entry.name,
                "方向": "多" if held["direction"] > 0 else "空",
                "手数": held["lots"], "开仓价": held["entry_price"],
                "当前止损": stop,
                "止损调整": "上移" if stop and held.get("stop") and stop > held["stop"] and held["direction"] > 0
                            else ("下移" if stop and held.get("stop") and stop < held["stop"] and held["direction"] < 0 else "维持"),
                "离场触发": plan.exit_trigger,
                "备注": plan.note,
            })
            continue

        if plan.direction != 0:
            # 系统口径下该品种有持仓，但用户账簿里没有 —— 这是"错过入场"，不是"不该做"
            stop_text = f"{plan.stop:.2f}" if np.isfinite(plan.stop) else "—"
            blocked.append({
                "代码": product_code, "名称": entry.name,
                "方向": "多" if plan.direction > 0 else "空",
                "当前价": round(price, 2),
                "系统止损": stop_text,
                "原因": (
                    f"系统口径下该品种已处于{'多' if plan.direction > 0 else '空'}头，"
                    f"而你的账簿里没有这笔持仓 —— 属于**错过入场**。"
                    f"**不要按现价追单**：突破策略的优势来自触发出场的那一刻，"
                    f"追进去等于用更差的价格承担同样宽度的止损（系统止损 {stop_text}）。"
                    f"正确做法是等下一次 {params.entry_window} 日通道突破信号重新入场。"
                ),
            })
            continue

        # 空仓：判断两个方向的突破是否值得挂单
        for direction, trigger, label in ((1, plan.entry_trigger, "向上突破做多"),
                                          (-1, plan.stop, "向下跌破做空")):
            if not np.isfinite(trigger):
                continue
            distance = abs(trigger - price) / price
            stop_distance = params.stop_atr * atr_now
            lots, detail = size_position(
                book.equity, risk, product, trigger, stop_distance,
                remaining_margin=remaining_margin, risk_multiplier=risk_multiplier,
            )
            order = {
                "代码": product_code, "名称": entry.name, "动作": label,
                "触发价": round(trigger, 2), "当前价": round(price, 2),
                "距离": distance, "方向": direction,
                "建议手数": lots,
                "初始止损": round(trigger - direction * stop_distance, 2),
                "单笔风险(元)": round(stop_distance * product.multiplier * lots, 2) if lots else 0.0,
                "预估往返成本(元)": round(product.round_trip_cost(trigger, lots), 2) if lots else 0.0,
                "保证金(元)": round(product.margin_per_lot(trigger) * lots, 2) if lots else 0.0,
                "执行时点": execution_window(target_day, product.night_session, last_bar_date),
                "可用": bool(lots > 0 and allowed),
                "不可用原因": "" if (lots > 0 and allowed) else (
                    "风险预算或保证金不足 1 手" if lots <= 0 else guard_note),
                "明细": detail,
            }
            new_orders.append(order)

    new_orders.sort(key=lambda item: (not item["可用"], item["距离"]))
    return {
        "target_day": target_day,
        "last_bar_date": last_bar_date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "book": asdict(book),
        "risk": asdict(risk),
        "params": params.label(),
        "guard": {"allowed": allowed, "note": guard_note, "risk_multiplier": risk_multiplier,
                  "remaining_margin": remaining_margin},
        "orders": new_orders,
        "manage": manage,
        "blocked": blocked,
        "market": market,
    }


def _used_margin(book: Book, products) -> float:
    """按建仓价估算账簿里在手持仓占用的保证金。"""
    total = 0.0
    for position in book.positions:
        product = products.get(position["code"])
        if product is None:
            continue
        price = float(position.get("entry_price") or product.ref_price)
        total += product.margin_per_lot(price) * position["lots"]
    return total


def render_markdown(plan: dict, products, notes: list[str]) -> str:
    """把方案渲染成人类可读、可打印的 Markdown 操作清单。"""
    target = plan["target_day"]
    book = plan["book"]
    guard = plan["guard"]
    lines = [
        f"# {target.isoformat()}（{_weekday_cn(target)}）交易计划",
        "",
        f"> 生成时间：{plan['generated_at']}　|　行情截止：{plan['last_bar_date'].isoformat()} 收盘　|"
        f"　策略：{plan['params']}",
        "",
        "> **免责声明**：本清单是比赛虚拟账户的决策辅助材料，不构成投资建议。"
        "它不下单、不接交易接口，所有价位都需要你在大赛专用交易软件里手工输入。",
        "",
    ]
    if notes:
        lines.extend(["## ⚠️ 先看这些提醒", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    lines.extend(_summary_section(plan))
    lines.extend(_orders_section(plan))
    lines.extend(_manage_section(plan))
    lines.extend(_market_section(plan))
    lines.extend(_risk_section(plan, book, guard))
    lines.extend(_schedule_section(plan, products))
    lines.extend(_blocked_section(plan))
    lines.extend(_review_section(target))
    return "\n".join(lines) + "\n"


def _summary_section(plan: dict) -> list[str]:
    """一、执行摘要：今天到底要做几件事。"""
    usable = [order for order in plan["orders"] if order["可用"]]
    closest = usable[:3]
    lines = ["## 一、执行摘要", ""]
    if not usable and not plan["manage"]:
        lines.append("- **今天没有需要新开的仓位，也没有在手持仓。保持空仓等待。**")
    if plan["manage"]:
        lines.append(f"- **必须挂出的保护性止损：{len(plan['manage'])} 个品种**"
                     "（见第三节，这是唯一不允许偷懒的动作）")
    if closest:
        lines.append(f"- **可以挂单等突破：{len(usable)} 个方向**，按距离由近到远：")
        for order in closest:
            lines.append(
                f"  - {order['名称']}（{order['代码']}）{order['动作']}："
                f"触发价 **{order['触发价']:.2f}**（距现价 {order['距离']:.2%}），"
                f"{order['建议手数']} 手，初始止损 {order['初始止损']:.2f}，"
                f"单笔风险约 {order['单笔风险(元)']:,.0f} 元"
            )
        if len(usable) > len(closest):
            lines.append(f"  - 另有 {len(usable) - len(closest)} 个更远的挂单，见第二节全表")
    if plan["blocked"]:
        lines.append(
            f"- **有 {len(plan['blocked'])} 个品种系统已经入场，但你的账簿没跟上**"
            "（属于「错过入场」，不要追单，原因见附节）"
        )
    if not plan["guard"]["allowed"]:
        lines.append(f"- ⛔ **今天不开新仓**：{plan['guard']['note']}")
    lines.append("")
    return lines


def _orders_section(plan: dict) -> list[str]:
    """二、新开仓挂单表。"""
    lines = ["## 二、新开仓挂单表", "",
             "按券商软件里的「条件单」理解：**价格触到就手动下单**。手数是按"
             "「单笔风险占权益固定比例」反推出来的，不是拍脑袋定的。", "",
             "> ⚠️ **同一品种出现多空两单时是二选一**：任一方向成交后，"
             "立刻撤销该品种另一方向的挂单，绝不允许双向同时持仓。", ""]
    if not plan["orders"]:
        lines.append("_今天没有任何品种给出入场信号。_")
        lines.append("")
        return lines
    lines.append("| 品种 | 方向 | 触发价 | 距现价 | 手数 | 初始止损 | 单笔风险 | 往返成本 | 占用保证金 | 可否执行 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for order in plan["orders"]:
        status = "✅ 可执行" if order["可用"] else f"⛔ {order['不可用原因']}"
        lines.append(
            f"| {order['名称']} {order['代码']} | {order['动作']} | {order['触发价']:.2f} "
            f"| {order['距离']:.2%} | {order['建议手数']} | {order['初始止损']:.2f} "
            f"| {order['单笔风险(元)']:,.0f} 元 | {order['预估往返成本(元)']:,.1f} 元 "
            f"| {order['保证金(元)']:,.0f} 元 | {status} |"
        )
    lines.extend(["", "**执行时点**（有夜盘的品种今晚 21:00 就能挂）：", ""])
    seen: set[str] = set()
    for order in plan["orders"]:
        if not order["可用"] or order["代码"] in seen:
            continue
        seen.add(order["代码"])
        lines.append(f"- {order['代码']}：{order['执行时点']}")
    lines.append("")
    return lines


def _blocked_section(plan: dict) -> list[str]:
    """把「系统有信号但没被采纳」的品种也说出来 —— 用户需要知道为什么今天不做这一单。"""
    if not plan["blocked"]:
        return []
    lines = ["## 附：今天不做的信号及原因", ""]
    lines.append("| 品种 | 系统方向 | 当前价 | 系统止损 | 为什么不做 |")
    lines.append("|---|---|---|---|---|")
    for item in plan["blocked"]:
        lines.append(
            f"| {item['名称']} {item['代码']} | {item['方向']} | {item['当前价']} "
            f"| {item['系统止损']} | {item['原因']} |"
        )
    lines.append("")
    return lines


def _manage_section(plan: dict) -> list[str]:
    """三、持仓管理：止损必须挂。"""
    lines = ["## 三、持仓管理与必须挂出的止损", ""]
    if not plan["manage"]:
        lines.append("_当前空仓，无需管理。_")
        lines.append("")
        return lines
    lines.append("| 品种 | 方向 | 手数 | 开仓价 | 止损价 | 止损调整 | 反向离场触发 |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in plan["manage"]:
        stop_text = f"{item['当前止损']:.2f}" if isinstance(item["当前止损"], (int, float)) and np.isfinite(item["当前止损"]) else "—"
        exit_text = f"{item['离场触发']:.2f}" if isinstance(item["离场触发"], (int, float)) and np.isfinite(item["离场触发"]) else "—"
        lines.append(
            f"| {item['名称']} {item['代码']} | {item['方向']} | {item['手数']} | {item['开仓价']} "
            f"| **{stop_text}** | {item['止损调整']} | {exit_text} |"
        )
    lines.extend([
        "",
        "规则：**止损价只能朝有利方向移动，永远不要朝不利方向调**。"
        "收盘价跌破（多单）/涨破（空单）反向离场触发价时，次一交易日开盘离场。",
        "",
    ])
    return lines


def _market_section(plan: dict) -> list[str]:
    """四、市场状态总览。"""
    lines = ["## 四、市场状态总览", "",
             "「状态」由 ADX 判定：≥25 趋势市，<20 震荡市。趋势市里突破策略胜率更高，"
             "震荡市里更容易被反复止损。", "",
             "| 品种 | 收盘 | ATR | ATR% | ADX | 状态 | 波动率分位 | 当前方向 |",
             "|---|---|---|---|---|---|---|---|"]
    for item in plan["market"]:
        direction = {1: "多", -1: "空", 0: "空仓"}.get(item["当前方向"], "空仓")
        percentile = f"{item['波动分位']:.0%}" if np.isfinite(item["波动分位"]) else "—"
        lines.append(
            f"| {item['名称']} {item['代码']} | {item['收盘']:.1f} | {item['ATR']:.1f} "
            f"| {item['ATR%']:.2%} | {item['ADX']:.1f} | {item['状态']} | {percentile} | {direction} |"
        )
    lines.append("")
    return lines


def _risk_section(plan: dict, book: dict, guard: dict) -> list[str]:
    """五、组合风控状态。"""
    drawdown = 0.0 if not book["peak_equity"] else max(0.0, (book["peak_equity"] - book["equity"]) / book["peak_equity"])
    return [
        "## 五、组合风控状态",
        "",
        "| 项目 | 数值 |",
        "|---|---|",
        f"| 当前权益 | {book['equity']:,.0f} 元 |",
        f"| 历史最高权益 | {book['peak_equity']:,.0f} 元 |",
        f"| 当前回撤 | {drawdown:.2%} |",
        f"| 风险倍数 | {guard['risk_multiplier']:.2f}× |",
        f"| 在手持仓 | {len(book['positions'])} 个 |",
        f"| 剩余保证金额度 | {guard['remaining_margin']:,.0f} 元 |",
        "",
        f"风控判定：{guard['note']}",
        "",
        "> 回撤触及暂停线时风险降至 1/4，而**不是完全停止交易** —— "
        "回测发现一旦停止开新仓，账户就失去唯一的恢复途径，暂停会被永久锁死。",
        "",
    ]


def _schedule_section(plan: dict, products) -> list[str]:
    """六、执行时刻表。"""
    target = plan["target_day"]
    night_start = plan["last_bar_date"].isoformat()
    lines = [
        "## 六、当天的盯盘时刻表",
        "",
        f"以下时间均为北京时间，对应交易日 **{target.isoformat()}**。",
        "",
        "| 时间 | 要做什么 |",
        "|---|---|",
        f"| {night_start} 20:55 | 打开交易软件，把第二节里**有夜盘**品种的条件单挂好 |",
        "| 21:00–23:00（夜盘） | 检查跳空是否直接穿越止损；触发的挂单成交后**立即补挂止损** |",
        "| 08:55 | 补挂无夜盘品种的条件单（苹果、红枣、花生等） |",
        "| 09:00–10:15 | 开盘半小时波动最大，**不要追价**，只按挂单价成交 |",
        "| 10:30–11:30 | 确认第三节的止损单都已在软件里 |",
        "| 13:30–14:50 | 盘中一般不操作；除非价格触及止损 |",
        "| 14:50–15:00 | 尾盘核对收盘价是否触发第三节的离场条件 |",
        "| 15:10 | 收盘后重新运行 `python scripts/daily_plan.py --refresh` 生成下一个交易日计划 |",
        "",
        "> **换月提醒**：临近交割月的合约必须在交割月前一交易日之前移仓或平仓，"
        "具体见本页末尾的提醒。",
        "",
    ]
    return lines


def _review_section(target: date) -> list[str]:
    """七、复盘模板：这是面试里最能体现「人做了什么」的部分。"""
    return [
        "## 七、收盘后复盘模板（请手工填写）",
        "",
        "> 这部分**故意留空**：面试官最关心的是「哪些判断是你做的」。"
        "把它填满，比让 AI 写十页代码更有说服力。",
        "",
        "```text",
        f"交易日：{target.isoformat()}",
        "1. 计划中的挂单，实际成交了哪些？为什么？",
        "   -",
        "2. 有没有偏离计划的操作（追价、提前平仓、改了止损）？当时的理由是什么？",
        "   -",
        "3. 如果亏了：是策略的正常成本，还是执行失误？",
        "   -",
        "4. 今天市场状态和计划里的判断一致吗？不一致的地方：",
        "   -",
        "5. 明天要不要调整参数或品种？如果调，依据是什么？",
        "   -",
        "```",
        "",
    ]


def main() -> None:
    """命令行入口：刷新行情 → 生成次日操作清单 → 写 Markdown + JSON。"""
    parser = argparse.ArgumentParser(description="生成下一个交易日的交易计划")
    parser.add_argument("--refresh", action="store_true", help="联网刷新行情缓存")
    parser.add_argument("--asof", help="指定目标交易日 YYYY-MM-DD（默认自动推断）")
    parser.add_argument("--tiers", default="core", help="品种池层级，逗号分隔")
    parser.add_argument("--equity", type=float, help="覆盖账户权益（元）")
    parser.add_argument("--risk", type=float, default=0.01, help="单笔风险占权益比例，默认 0.01")
    parser.add_argument("--max-positions", type=int, default=8, help="最多同时持有品种数")
    parser.add_argument("--derive", action="store_true", help="用回测口径的系统组合代替手工账簿")
    parser.add_argument("--state", default=str(STATE_PATH), help="账簿文件路径")
    parser.add_argument("--out", default=str(REPORT_DIR), help="计划输出目录")
    parser.add_argument("--suffix", default="", help="输出文件名后缀，例如 -tracked")
    args = parser.parse_args()

    entries = load_universe(tiers=tuple(args.tiers.split(",")))
    products = load_products()
    frames: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    for entry in entries:
        frame = load_daily(entry.continuous, refresh=args.refresh)
        frames[entry.continuous] = frame
        warning = stale_warning(frame)
        if warning:
            notes.append(f"{entry.code}：{warning}")
        report = quality_report(entry.continuous, frame)
        if not report["ok"]:
            notes.append(f"{entry.code}：数据质量检查未通过 —— {'；'.join(report['errors'])}")

    params = DEFAULT_STRATEGY
    features = {code: compute_features(frame, {"atr": params.atr_window, "fast": params.entry_window,
                                               "slow": 50, "donchian": params.entry_window,
                                               "exit": params.exit_window, "adx": 14, "boll": 20})
                for code, frame in frames.items()}
    states = {}
    for code in frames:
        state_frame, _, _ = breakout_positions(features[code], params, code)
        states[code] = state_frame

    state_path = Path(args.state)
    if args.derive:
        prepared = prepare_signals(frames, products, params)
        result = run_backtest(prepared, RiskConfig(risk_per_trade=args.risk), name="系统跟踪组合",
                              params_note=params.label())
        book = book_from_backtest(result)
        notes.append("本次使用 **--derive**：持仓来自回测口径的系统跟踪组合，不代表真实成交。")
    else:
        book = load_book(state_path)
        if args.equity:
            book.equity = args.equity
            book.peak_equity = max(book.peak_equity, args.equity)

    if book.equity > book.peak_equity:
        book.peak_equity = book.equity

    risk = RiskConfig(equity=book.equity, risk_per_trade=args.risk, max_positions=args.max_positions)

    now = datetime.now()
    if args.asof:
        target = datetime.strptime(args.asof, "%Y-%m-%d").date()
        last_bar_date = max(
            pd.to_datetime(last_completed_bar(frame, now)["date"]).date() for frame in frames.values()
        )
        target = datetime.strptime(args.asof, "%Y-%m-%d").date()
    else:
        target, last_bar_date = estimate_next_trading_day(frames, now)
        if target <= now.date():
            notes.append(
                f"**这份清单针对的交易日 {target.isoformat()} 已经开始或已过半程**："
                "当前时间还没到 15:00，当天日线尚未收盘，所以依据的是上一根已收盘 K 线。"
                "建议每日 15:10 之后重新运行，才有完整意义。"
            )

    plan = build_plan(entries, products, frames, features, states, book, risk, params,
                      target, last_bar_date)

    for line in roll_warnings(products):
        if any(entry.code in line for entry in entries):
            notes.append("换月提醒：" + line)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(plan, products, notes)
    md_path = out_dir / f"{target.isoformat()}{args.suffix}.md"
    json_path = out_dir / f"{target.isoformat()}{args.suffix}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(markdown)
    print(f"已写出：{md_path}")
    print(f"已写出：{json_path}")


if __name__ == "__main__":
    main()
