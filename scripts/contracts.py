"""郑商所品种与合约规格层：合约乘数、最小变动价位、手续费、保证金、主力合约与换月。

数据来源与可信度
- 联网快照：AkShare ``futures_fees_info``（聚合公开行情商的交易所标准数据）。
- 离线快照：``config/czce_products.json``，由本模块生成并提交进仓库，断网可跑。
- **该快照必须与郑商所官网公布的手续费/保证金标准核对后才可用于真实决策。**
- 赛制按交易所标准 1.5 倍收取手续费，见 :data:`FEE_MULTIPLIER`。

手续费建模的关键坑
``futures_fees_info`` 的"费率"列对**固定收费**品种是占位值（0.000001），只有
"1手开仓费用/1手平今费用"（元/手）才可信；对**按成交额收费**品种（甲醇、纯碱、
尿素、烧碱、PX、瓶片）则相反，"费率"列可信而"费用/手"列是错的。
本模块用 :data:`RATE_EPS` 自动判别两类，避免回测低估或高估成本。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "config" / "czce_products.json"

FEE_MULTIPLIER = 1.5
"""赛制：手续费按交易所标准 1.5 倍收取。"""

SLIPPAGE_TICKS = 1.0
"""成交滑点假设：每次成交按 1 个最小变动价位计（保守）。"""

MARGIN_BUFFER = 1.3
"""保证金安全垫：按交易所标准的 1.3 倍估算占用，防止节前/临近交割提保导致强平。"""

RATE_EPS = 1.1e-6
""""费率"字段低于该值即视为占位值，手续费改用固定元/手。"""

NIGHT_SESSION = frozenset(
    {"CF", "CY", "SR", "TA", "MA", "RM", "OI", "FG", "SA", "UR", "PF", "SH", "PX", "PR", "SF", "SM"}
)
"""有夜盘（21:00-23:00）的郑商所品种；苹果、红枣、花生、动力煤及稻麦类无夜盘。"""

SESSIONS: dict[str, tuple[str, str]] = {
    "night": ("21:00", "23:00"),
    "morning_a": ("09:00", "10:15"),
    "morning_b": ("10:30", "11:30"),
    "afternoon": ("13:30", "15:00"),
}
"""交易时段（北京时间）。夜盘属于**下一个**交易日。"""


def parse_delivery(code: str, today: date) -> str | None:
    """把 CZCE 合约代码（如 CF701）解析成到期年月 ``YYYY-MM``；连续合约返回 None。

    CZCE 用一位年份 + 两位月份，需要按当前年代还原四位年份。
    """
    match = re.fullmatch(r"[A-Za-z]{1,2}(\d)(\d{2})", code.strip())
    if not match:
        return None
    year_digit, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    year = (today.year // 10) * 10 + year_digit
    if year < today.year - 1:
        year += 10
    return f"{year:04d}-{month:02d}"


def months_until(delivery_ym: str, today: date) -> int:
    """到期年月距今还有几个月（可为负）。"""
    year, month = (int(part) for part in delivery_ym.split("-"))
    return (year - today.year) * 12 + (month - today.month)


@dataclass(frozen=True)
class Product:
    """单个郑商所品种的规格与成本参数。金额单位一律为元。"""

    code: str
    name: str
    multiplier: int
    tick: float
    margin_rate: float
    ref_price: float
    main_contract: str
    main_open_interest: float
    delivery_ym: str | None
    night_session: bool
    fee_open_rate: float
    fee_open_fixed: float
    fee_close_rate: float
    fee_close_fixed: float
    fee_today_rate: float
    fee_today_fixed: float
    snapshot_date: str

    def fee_per_lot(self, price: float, *, opening: bool = True, close_today: bool = False) -> float:
        """按交易所标准计算的单边手续费（元/手），**不含**赛制 1.5 倍。"""
        if opening:
            rate, fixed = self.fee_open_rate, self.fee_open_fixed
        elif close_today:
            rate, fixed = self.fee_today_rate, self.fee_today_fixed
        else:
            rate, fixed = self.fee_close_rate, self.fee_close_fixed
        return fixed + rate * price * self.multiplier

    def charged_fee(self, price: float, lots: float, *, opening: bool, close_today: bool = False) -> float:
        """按赛制 1.5 倍收取的手续费（元）。"""
        return self.fee_per_lot(price, opening=opening, close_today=close_today) * lots * FEE_MULTIPLIER

    def slippage(self, lots: float) -> float:
        """单边滑点成本（元）。"""
        return self.tick * SLIPPAGE_TICKS * self.multiplier * lots

    def round_trip_cost(self, price: float, lots: float, *, close_today: bool = False) -> float:
        """一次开平的总成本（元）：1.5 倍手续费双边 + 双边滑点。"""
        opening = self.charged_fee(price, lots, opening=True)
        closing = self.charged_fee(price, lots, opening=False, close_today=close_today)
        return opening + closing + self.slippage(lots) * 2

    def margin_per_lot(self, price: float) -> float:
        """单手持仓保证金估算（元），含 :data:`MARGIN_BUFFER` 安全垫。"""
        return price * self.multiplier * self.margin_rate * MARGIN_BUFFER

    def notional_per_lot(self, price: float) -> float:
        """单手合约面值（元）。"""
        return price * self.multiplier


def _pick_main(rows: list[dict[str, Any]], today: date) -> dict[str, Any]:
    """在候选合约里挑主力：优先到期>=2 个月的合约中持仓量最大者。

    "交割月前一交易日停止交易"是赛制硬约束，因此把只剩 1 个月以内到期的合约排除在主力之外，
    但仍会保留它们的数据，方便给出换月提醒。
    """
    scored = []
    for row in rows:
        delivery = parse_delivery(str(row["合约代码"]), today)
        if delivery is None:
            continue
        scored.append((months_until(delivery, today), row, delivery))
    if not scored:
        raise ValueError("没有可解析到期的合约")
    safe = [item for item in scored if item[0] >= 2 and float(item[1].get("持仓量") or 0) > 0]
    pool = safe or [item for item in scored if float(item[1].get("持仓量") or 0) > 0] or scored
    gap, row, delivery = max(pool, key=lambda item: float(item[1].get("持仓量") or 0))
    return {"row": row, "delivery_ym": delivery, "months": gap}


def _split_fee(rate: Any, fixed: Any) -> tuple[float, float]:
    """按 :data:`RATE_EPS` 判别比例收费还是固定收费，返回 (rate, fixed)。"""
    rate_value = float(rate or 0.0)
    fixed_value = float(fixed or 0.0)
    if rate_value > RATE_EPS:
        return rate_value, 0.0
    return 0.0, fixed_value


def fetch_products(today: date | None = None) -> dict[str, dict[str, Any]]:
    """联网拉取郑商所合约规格，返回 ``{品种代码: 规格字典}``。"""
    import akshare as ak

    today = today or date.today()
    frame = ak.futures_fees_info()
    czce = frame[frame["交易所"] == "CZCE"]
    if czce.empty:
        raise ValueError("AkShare 未返回郑商所合约")
    products: dict[str, dict[str, Any]] = {}
    for code, group in czce.groupby("品种代码"):
        rows = group.to_dict("records")
        try:
            main = _pick_main(rows, today)
        except ValueError:
            continue
        row = main["row"]
        open_rate, open_fixed = _split_fee(row.get("开仓费率"), row.get("1手开仓费用"))
        close_rate, close_fixed = _split_fee(row.get("平仓费率"), row.get("1手平仓费用"))
        today_rate, today_fixed = _split_fee(row.get("平今费率"), row.get("1手平今费用"))
        code_str = str(code)
        products[code_str] = {
            "code": code_str,
            "name": str(row.get("品种名称") or code_str),
            "multiplier": int(float(row.get("合约乘数") or 1)),
            "tick": float(row.get("最小跳动") or 1.0),
            "margin_rate": float(row.get("做多保证金率") or 0.1),
            "ref_price": float(row.get("最新价") or 0.0),
            "main_contract": str(row.get("合约代码")),
            "main_open_interest": float(row.get("持仓量") or 0.0),
            "delivery_ym": main["delivery_ym"],
            "night_session": code_str in NIGHT_SESSION,
            "fee_open_rate": open_rate,
            "fee_open_fixed": open_fixed,
            "fee_close_rate": close_rate,
            "fee_close_fixed": close_fixed,
            "fee_today_rate": today_rate,
            "fee_today_fixed": today_fixed,
            "snapshot_date": today.isoformat(),
        }
    return products


def save_snapshot(products: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    """把规格快照写成可读 JSON 并放进仓库。"""
    target = path or SNAPSHOT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_note": (
            "郑商所品种规格快照，由 scripts/contracts.py 生成。"
            "手续费/保证金为交易所标准，赛制按 1.5 倍收取；使用前必须与郑商所官网核对。"
        ),
        "products": products,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_products(refresh: bool = False, today: date | None = None) -> dict[str, Product]:
    """读取品种规格：默认用本地快照，``refresh=True`` 时联网刷新并写回快照。"""
    if refresh or not SNAPSHOT_PATH.exists():
        products = fetch_products(today)
        if products:
            save_snapshot(products)
    else:
        payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        products = payload.get("products", payload)
    return {code: Product(**spec) for code, spec in products.items()}


def roll_warnings(products: dict[str, Product], today: date | None = None, warn_months: int = 2) -> list[str]:
    """给出临近交割月的换月提醒（赛制：交割月前一交易日停止交易）。"""
    today = today or date.today()
    out: list[str] = []
    for product in products.values():
        if not product.delivery_ym:
            continue
        gap = months_until(product.delivery_ym, today)
        if gap <= warn_months:
            out.append(
                f"{product.code} {product.name} 主力 {product.main_contract}"
                f"（到期 {product.delivery_ym}，距今 {gap} 个月）—— 需在交割月前一交易日之前换月或平仓"
            )
    return sorted(out)


def main() -> None:
    """命令行入口：刷新品种规格快照并打印概览。"""
    import argparse

    parser = argparse.ArgumentParser(description="刷新郑商所品种规格快照")
    parser.add_argument("--offline", action="store_true", help="只读取本地快照，不联网")
    args = parser.parse_args()
    products = load_products(refresh=not args.offline)
    print(f"品种数: {len(products)}  快照: {SNAPSHOT_PATH}")
    print(f"{'代码':<4}{'名称':<10}{'乘数':>4}{'tick':>6}{'保证金':>8}{'开仓(元)':>10}{'平今(元)':>10}  主力合约")
    for product in sorted(products.values(), key=lambda item: item.code):
        print(
            f"{product.code:<4}{product.name:<10}{product.multiplier:>4}{product.tick:>6.1f}"
            f"{product.margin_rate:>8.2%}{product.fee_per_lot(product.ref_price):>10.2f}"
            f"{product.fee_per_lot(product.ref_price, opening=False, close_today=True):>10.2f}"
            f"  {product.main_contract}"
        )
    warnings = roll_warnings(products)
    if warnings:
        print("\n换月提醒：")
        for line in warnings:
            print("  -", line)


if __name__ == "__main__":
    main()
