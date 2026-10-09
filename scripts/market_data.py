"""行情数据层：郑商所日线/分钟线的获取、本地缓存与基础质量校验。

设计原则
- **只用新浪接口**（AkShare ``futures_zh_daily_sina`` / ``futures_zh_minute_sina``），
  它是唯一能给出 2005 年以来长历史的免费源。
- 抓到的原始数据一律落盘到 ``data/cache/``（被 .gitignore 忽略），
  联网失败时退回本地缓存并**显式标注数据年龄**，绝不静默当作最新数据。
- 日线的"开盘价"是**当日 21:00 夜盘开盘**（夜盘属于下一交易日），
  因此 T 日收盘算出的信号可以在 T 日 21:00 就执行，这一点对操作清单很关键。

已知局限（会写进质量报告，不掩盖）
- 新浪连续合约（CF0）是主力拼接，**换月处有价格跳空**，回测会因此产生噪声。
- 连续合约不含具体合约的到期约束，真实交易需按主力合约代码下单。
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"

DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "open_interest"]
MINUTE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def _cache_path(kind: str, code: str, period: str = "") -> Path:
    suffix = f"_{period}" if period else ""
    return CACHE_DIR / f"{kind}_{code}{suffix}.csv"


def clean_bars(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """剔除不可用于计算的 bar，并返回剔除统计。

    新浪连续合约的历史里有两类脏数据，都会污染 ATR、通道和止损模拟：
    1. **高低价逻辑越界**：``open`` 落在 ``[low, high]`` 之外（2006-2008 年偶发）。
    2. **节假日填充**：国庆/春节停市期间，接口把上一根 bar 原样复制到每一天
       （例如 CF0 在 2005-10-03 至 2005-10-12 连续 8 根完全相同）。
       这类"假交易日"会让滚动窗口多算好几天的陈旧价格。
    只删除，不做任何插值或修正，保证回测用的每一根 bar 都来自真实成交。
    """
    out = frame.copy()
    high, low = out["high"], out["low"]
    bad_logic = (high < out[["open", "close"]].max(axis=1)) | (low > out[["open", "close"]].min(axis=1))
    price_cols = ["open", "high", "low", "close", "volume"]
    duplicate = out[price_cols].eq(out[price_cols].shift()).all(axis=1)
    mask = ~(bad_logic | duplicate)
    stats = {"剔除_高低价越界": int(bad_logic.sum()), "剔除_重复填充": int(duplicate.sum())}
    cleaned = out[mask].reset_index(drop=True)
    cleaned.attrs["cleaning"] = stats
    cleaned.attrs["rows_before_cleaning"] = len(out)
    return cleaned, stats


def _standardize(frame: pd.DataFrame, mapping: dict[str, str], columns: list[str], time_column: str) -> pd.DataFrame:
    """统一列名、类型与顺序，并剔除无法解析的行。"""
    renamed = frame.rename(columns=mapping).copy()
    missing = [column for column in columns if column not in renamed.columns]
    if missing:
        raise ValueError(f"行情缺少字段: {missing}")
    renamed["open_interest"] = renamed.get("open_interest", np.nan)
    out = renamed[columns].copy()
    out[time_column] = pd.to_datetime(out[time_column], errors="coerce")
    for column in columns:
        if column != time_column:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=[time_column])
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
    return out


def fetch_daily(code: str) -> pd.DataFrame:
    """联网抓取日线长历史（2005 年起）。"""
    import akshare as ak

    raw = ak.futures_zh_daily_sina(symbol=code)
    if raw is None or raw.empty:
        raise ValueError(f"{code} 日线返回空数据")
    return _standardize(
        raw,
        {"date": "date", "open": "open", "high": "high", "low": "low",
         "close": "close", "volume": "volume", "hold": "open_interest"},
        DAILY_COLUMNS,
        "date",
    )


def fetch_minute(code: str, period: str = "30") -> pd.DataFrame:
    """联网抓取分钟线（新浪只提供最近约 1000 根）。"""
    import akshare as ak

    raw = ak.futures_zh_minute_sina(symbol=code, period=period)
    if raw is None or raw.empty:
        raise ValueError(f"{code} {period} 分钟线返回空数据")
    return _standardize(
        raw,
        {"datetime": "datetime", "时间": "datetime", "日期": "datetime", "open": "open",
         "high": "high", "low": "low", "close": "close", "volume": "volume"},
        MINUTE_COLUMNS,
        "datetime",
    )


def load_daily(code: str, *, refresh: bool = False, days: int | None = None, clean: bool = True) -> pd.DataFrame:
    """读取日线：优先缓存，``refresh=True`` 时联网更新缓存；联网失败退回缓存并标注。

    ``clean=True``（默认）会剔除 :func:`clean_bars` 识别出的坏 bar；统计挂在
    ``frame.attrs['cleaning']`` 上，供质量报告引用。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path("daily", code)
    frame: pd.DataFrame | None = None
    if refresh or not path.exists():
        try:
            frame = fetch_daily(code)
            frame.to_csv(path, index=False)
        except Exception as exc:  # 联网失败必须可见，但不能让整个流程崩掉
            if not path.exists():
                raise RuntimeError(f"{code} 日线获取失败且无本地缓存: {type(exc).__name__}") from exc
            print(f"[警告] {code} 日线抓取失败（{type(exc).__name__}），改用本地缓存", file=sys.stderr)
    if frame is None:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
    if clean:
        frame, _ = clean_bars(frame)
    if days:
        frame = frame.tail(days).reset_index(drop=True)
    return frame


def load_minute(code: str, period: str = "30", *, refresh: bool = False) -> pd.DataFrame:
    """读取分钟线：优先缓存，``refresh=True`` 时联网更新。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path("minute", code, period)
    frame: pd.DataFrame | None = None
    if refresh or not path.exists():
        try:
            frame = fetch_minute(code, period)
            frame.to_csv(path, index=False)
        except Exception as exc:
            if not path.exists():
                raise RuntimeError(f"{code} 分钟线获取失败且无本地缓存: {type(exc).__name__}") from exc
            print(f"[警告] {code} 分钟线抓取失败（{type(exc).__name__}），改用本地缓存", file=sys.stderr)
    if frame is None:
        frame = pd.read_csv(path)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame


def quality_report(code: str, frame: pd.DataFrame, time_column: str = "date") -> dict[str, Any]:
    """基础质量校验：行数、时间范围、重复、缺口、价格逻辑、极端跳空（换月痕迹）。"""
    report: dict[str, Any] = {"code": code, "rows": int(len(frame)), "errors": [], "warnings": []}
    cleaning = frame.attrs.get("cleaning") if hasattr(frame, "attrs") else None
    if cleaning:
        removed = sum(cleaning.values())
        report["cleaning"] = cleaning
        report["rows_before_cleaning"] = frame.attrs.get("rows_before_cleaning", len(frame) + removed)
        if removed:
            report["warnings"].append(
                "已剔除 " + str(removed) + " 根脏 bar（"
                + "、".join(f"{key.replace('剔除_', '')} {value} 根" for key, value in cleaning.items() if value)
                + "）"
            )
    if frame.empty:
        report["errors"].append("没有数据行")
        return report
    times = pd.to_datetime(frame[time_column])
    report["start"] = str(times.min().date() if time_column == "date" else times.min())
    report["end"] = str(times.max().date() if time_column == "date" else times.max())
    if times.duplicated().any():
        report["errors"].append(f"重复时间 {int(times.duplicated().sum())} 行")
    if not times.is_monotonic_increasing:
        report["errors"].append("时间未升序")
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all():
        report["errors"].append("存在缺失或非有限价格")
    bad = (frame["high"] < frame[["open", "close"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close"]].min(axis=1)
    )
    if bool(bad.any()):
        report["errors"].append(f"高低价逻辑错误 {int(bad.sum())} 行")
    if (frame[["open", "high", "low", "close"]] <= 0).to_numpy().any():
        report["errors"].append("存在非正价格")
    returns = frame["close"].pct_change()
    jumps = returns[returns.abs() > 0.04]
    report["jump_days"] = int(len(jumps))
    report["max_abs_return"] = float(returns.abs().max()) if len(returns.dropna()) else 0.0
    if report["jump_days"]:
        report["warnings"].append(
            f"|单日涨跌| > 4% 共 {report['jump_days']} 天（含真实极端行情与连续合约换月跳空）"
        )
    if time_column == "date" and len(frame) > 1:
        gaps = times.diff().dt.days.dropna()
        report["max_calendar_gap_days"] = int(gaps.max())
        if gaps.max() > 20:
            report["warnings"].append(f"最长日历间隔 {int(gaps.max())} 天，需核对是否长期停牌或数据缺失")
    report["ok"] = not report["errors"]
    return report


def last_completed_bar(frame: pd.DataFrame, now: datetime | None = None) -> pd.Series:
    """取最近一根**已收盘**的日线。

    日盘的收盘时间是 15:00。若当前时间未到 15:00，则当天的日线还在形成中，
    必须退回上一根，避免把未完成的行情当成信号依据。
    """
    now = now or datetime.now()
    if frame.empty:
        raise ValueError("行情为空")
    last = frame.iloc[-1]
    last_date = pd.to_datetime(last["date"])
    if last_date.date() == now.date() and now.time() < datetime.strptime("15:00", "%H:%M").time():
        if len(frame) < 2:
            raise ValueError("只有一根未收盘的日线")
        return frame.iloc[-2]
    return last


def trading_days_between(frame: pd.DataFrame, start: date, end: date) -> int:
    """统计区间内已有的交易日数量（用实际数据当交易日历）。"""
    times = pd.to_datetime(frame["date"]).dt.date
    return int(((times >= start) & (times <= end)).sum())


def next_session_note(product_code: str, has_night: bool, asof: date) -> str:
    """给出执行时点的中文说明：有夜盘的品种当日晚 21:00 即可执行。"""
    if has_night:
        return f"{asof.isoformat()} 21:00 夜盘开盘（夜盘属于下一交易日）"
    return f"下一交易日 09:00 日盘开盘（{product_code} 无夜盘）"


def summarize(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把多个品种的行情概况汇总成一张表，便于打印与写报告。"""
    rows = []
    for code, frame in frames.items():
        last = frame.iloc[-1]
        rows.append(
            {
                "代码": code,
                "行数": len(frame),
                "起始": pd.to_datetime(frame["date"].iloc[0]).date(),
                "最新": pd.to_datetime(last["date"]).date(),
                "收盘": float(last["close"]),
                "20日涨跌": float(last["close"] / frame["close"].iloc[-21] - 1) if len(frame) > 21 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stale_warning(frame: pd.DataFrame, max_age_days: int = 5, now: datetime | None = None) -> str | None:
    """数据陈旧提醒：最新一根日线距今天超过 N 个自然日就告警。"""
    now = now or datetime.now()
    last = pd.to_datetime(frame["date"].iloc[-1]).date()
    age = (now.date() - last).days
    if age > max_age_days:
        return f"最新日线 {last} 距今 {age} 天，可能因节假日或抓取失败，使用前请确认"
    return None


def expected_next_trading_day(frame: pd.DataFrame, now: datetime | None = None) -> date:
    """推断下一个交易日的日期：取最后一根已收盘日线之后的第一个工作日（粗略，不含法定节假日）。"""
    now = now or datetime.now()
    last = pd.to_datetime(last_completed_bar(frame, now)["date"]).date()
    candidate = max(last, now.date()) + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="行情缓存刷新与质量检查")
    parser.add_argument("--symbols", default="CF0,SR0,TA0,MA0,RM0,OI0,FG0,SA0")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    codes = [item.strip() for item in args.symbols.split(",") if item.strip()]
    for code in codes:
        frame = load_daily(code, refresh=args.refresh)
        report = quality_report(code, frame)
        flag = "OK" if report["ok"] else "FAIL"
        print(f"[{flag}] {code:<6} 行数={report['rows']:<6} {report['start']} -> {report['end']} "
              f"跳空天数={report['jump_days']}")
        for warning in report["warnings"]:
            print(f"       ! {warning}")
