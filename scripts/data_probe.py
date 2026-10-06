"""行情数据探针：获取、缓存并检查 30 分钟期货数据。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
ROOT = Path(__file__).resolve().parents[1]


def fetch_akshare(symbol: str, period: str, limit: int) -> pd.DataFrame:
    """调用 AkShare 的新浪期货分钟接口，返回统一字段。"""
    import akshare as ak

    raw = ak.futures_zh_minute_sina(symbol=symbol, period=period)
    if raw is None or raw.empty:
        raise ValueError("AkShare 返回空数据")
    renamed = raw.rename(
        columns={
            "时间": "datetime",
            "日期": "datetime",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"AkShare 缺少字段: {missing}")
    return renamed[REQUIRED_COLUMNS].tail(limit).copy()


def make_demo_data(limit: int, seed: int = 42) -> pd.DataFrame:
    """生成仅用于流程测试的模拟行情，并在报告中明确标注。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-02 09:00", periods=limit, freq="30min")
    close = 6_000 + np.cumsum(rng.normal(0, 12, limit))
    opening = close - rng.normal(0, 5, limit)
    spread = rng.uniform(2, 12, limit)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": opening,
            "high": np.maximum(close, opening) + spread,
            "low": np.minimum(close, opening) - spread,
            "close": close,
            "volume": rng.integers(100, 3_000, limit),
        }
    )


def validate_data(data: pd.DataFrame) -> dict[str, Any]:
    """检查字段、时间顺序、重复记录、空值和价格逻辑。"""
    result: dict[str, Any] = {"rows": int(len(data)), "errors": [], "warnings": [], "quality": "失败"}
    if data.empty:
        result["errors"].append("没有数据行")
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        result["errors"].append(f"缺少字段: {missing}")
        return result
    times = pd.to_datetime(data["datetime"], errors="coerce")
    if times.isna().any():
        result["errors"].append(f"无效时间: {int(times.isna().sum())} 行")
    if times.duplicated().any():
        result["errors"].append(f"重复时间: {int(times.duplicated().sum())} 行")
    if not times.is_monotonic_increasing:
        result["errors"].append("时间未按升序排列")
    numeric = ["open", "high", "low", "close", "volume"]
    numbers = data[numeric].apply(pd.to_numeric, errors="coerce")
    missing_values = int((~np.isfinite(numbers.to_numpy(dtype=float))).sum())
    if missing_values:
        result["errors"].append(f"缺失、非数值或非有限数值: {missing_values} 个")
    bad_bars = (numbers["high"] < numbers[["open", "close"]].max(axis=1)) | (
        numbers["low"] > numbers[["open", "close"]].min(axis=1)
    )
    if bad_bars.any():
        result["errors"].append(f"价格逻辑错误: {int(bad_bars.sum())} 行")
    if (numbers[["open", "high", "low", "close"]] <= 0).any().any() or (numbers["volume"] < 0).any():
        result["errors"].append("存在非正价格或负成交量")
    if len(data) < 100:
        result["warnings"].append("样本少于 100 根，只适合流程测试")
    result["warnings"].append("未核验交易日历、缺失K线、换月和实时性；不能据此计算完整率")
    result["start"] = str(times.min()) if len(data) else None
    result["end"] = str(times.max()) if len(data) else None
    result["quality"] = "通过" if not result["errors"] else "失败"
    return result


def write_report(path: Path, source: str, symbol: str, period: str, check: dict[str, Any]) -> None:
    """把数据来源和检查结果写成可审阅的 Markdown 报告。"""
    lines = [
        "# 数据探针报告",
        "",
        f"- 数据来源：{source}",
        f"- 检查时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        f"- 品种：{symbol}",
        f"- 周期：{period} 分钟",
        f"- 样本数：{check.get('rows', 0)}",
        f"- 时间范围：{check.get('start')} 至 {check.get('end')}",
        f"- 基础字段检查：**{check.get('quality')}**",
        "",
        "## 错误",
        "",
    ]
    lines.extend(f"- {item}" for item in check["errors"] or ["无"])
    lines.extend(["", "## 警告", ""])
    lines.extend(f"- {item}" for item in check["warnings"] or ["无"])
    lines.extend(["", "> CF0 是数据接口连续合约示例，不是可直接交易的具体合约。",
                  "> 模拟数据含非交易时段，只用于流程测试，不得用于收益结论。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """解析命令行参数，完成获取、保存和质量报告。"""
    parser = argparse.ArgumentParser(description="30 分钟行情数据探针")
    parser.add_argument("--symbol", default="CF0", help="AkShare 品种代码，例如 CF0")
    parser.add_argument("--period", default="30", choices=["1", "5", "15", "30", "60"])
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output-dir", default="data/probe")
    parser.add_argument("--demo", action="store_true", help="强制使用模拟数据")
    parser.add_argument("--input", type=Path, help="离线检查本地 CSV，不联网")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10000 or not re.fullmatch(r"[A-Za-z0-9]+", args.symbol):
        parser.error("limit 必须在 1 到 10000 之间，symbol 只能包含英文字母和数字")
    if args.demo and args.input:
        parser.error("--demo 和 --input 不能同时使用")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source = "模拟数据（流程测试）"
    mode = "demo"
    if args.input:
        data = pd.read_csv(args.input)
        source, mode = "本地 CSV（真实性与来源需自行核验）", "local"
    elif args.demo:
        data = make_demo_data(args.limit)
    else:
        try:
            # 子进程限制整个抓取时间，避免第三方接口一直挂起。
            worker = "import sys; from data_probe import fetch_akshare; print(fetch_akshare(sys.argv[1], sys.argv[2], int(sys.argv[3])).to_json(orient='split'))"
            fetched = subprocess.run([sys.executable, "-c", worker, args.symbol, args.period, str(args.limit)],
                                     cwd=Path(__file__).parent, capture_output=True, text=True,
                                     encoding="utf-8", timeout=30, check=True)
            payload = json.loads(fetched.stdout)
            data = pd.DataFrame(payload["data"], columns=payload["columns"])
            source = "AkShare futures_zh_minute_sina"
            mode = "real"
        except Exception as exc:  # 网络和第三方接口错误必须转为可见的降级状态
            source += f"；抓取降级原因：{type(exc).__name__}"
            print(f"AkShare 获取失败，转用模拟数据：{type(exc).__name__}")
            data = make_demo_data(args.limit)
    check = validate_data(data)
    csv_path = output_dir / f"{args.symbol}_{args.period}m_{mode}.csv"
    report_path = output_dir / f"quality_report_{mode}.md"
    write_report(report_path, source, args.symbol, args.period, check)
    if check["errors"]:
        print("数据校验失败：" + "；".join(check["errors"]))
        print(f"报告: {report_path}")
        raise SystemExit(1)
    data.to_csv(csv_path, index=False)
    print(f"数据来源: {source}")
    print(f"样本数: {check['rows']}")
    print(f"质量检查: {check['quality']}")
    print(f"时间范围: {check['start']} 至 {check['end']}")
    print("注意: 仅通过基础检查；完整率、实时性、连续合约换月尚未核验")
    print(f"CSV: {csv_path}")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
