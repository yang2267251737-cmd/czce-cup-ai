"""技术指标层：只用 pandas / numpy 实现，不含任何未来函数。

约定
- 所有函数接收 Series/DataFrame，返回**与输入同索引**的 Series/DataFrame。
- 一律使用"截止当前 bar 为止"的滚动窗口（``rolling``/``ewm``），不使用中心窗口或整段统计，
  确保在历史上任意一天得到的值，与当天真实能看到的值一致（无前视偏差）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均。"""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均。"""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅 TR = max(H-L, |H-C_prev|, |L-C_prev|)。"""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """平均真实波幅（Wilder 平滑），仓位与止损的统一尺度。"""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def donchian(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """唐奇安通道。**通道不含当日**（用 ``shift(1)``），否则当日最高价必然等于通道上轨。"""
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    return upper, lower


def bollinger_z(close: pd.Series, window: int = 20) -> pd.Series:
    """收盘价相对布林带中轨的 z 分数，用于均值回归信号。"""
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return (close - mean) / std.replace(0.0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """平均趋向指数：>25 视为有趋势，<20 视为震荡。"""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr = true_range(high, low, close).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / tr.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / tr.replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """考夫曼效率比：净位移 / 总路程。接近 1 是流畅趋势，接近 0 是震荡。"""
    change = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window, min_periods=window).sum()
    return (change / volatility.replace(0.0, np.nan)).clip(0.0, 1.0)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """年化历史波动率（按 244 个交易日）。"""
    return close.pct_change().rolling(window, min_periods=window).std(ddof=0) * np.sqrt(244)


def slope(series: pd.Series, window: int = 20) -> pd.Series:
    """滚动线性回归斜率，除以价格水平做标准化（趋势强度，无量纲）。

    这里是**向量化**实现，不是逐窗 polyfit：窗口内自变量固定为 0..w-1，
    因此斜率 = (E[xy] - E[x]E[y]) / Var(x)，其中 E[x]、Var(x) 是常数，
    E[xy] 可由"滚动加权和"展开成两个普通滚动和。5000 根日线从十几秒降到毫秒级。
    """
    values = series.astype(float)
    mean_x = (window - 1) / 2.0
    var_x = (window * window - 1) / 12.0
    positions = pd.Series(np.arange(len(values), dtype=float), index=values.index)
    weighted = values * positions
    sum_y = values.rolling(window, min_periods=window).sum()
    sum_xy = weighted.rolling(window, min_periods=window).sum()
    # 把全局位置还原成窗口内位置 0..w-1
    shifted_positions = positions - (window - 1)
    mean_xy = (sum_xy - shifted_positions * sum_y) / window
    mean_y = sum_y / window
    raw = (mean_xy - mean_x * mean_y) / var_x
    return raw / values.replace(0.0, np.nan)


def percentile_rank(series: pd.Series, window: int = 244) -> pd.Series:
    """当前值在过去 window 天中的分位数（0-1），用于波动率/价格位置判断。"""
    return series.rolling(window, min_periods=max(20, window // 4)).rank(pct=True)


def compute_features(frame: pd.DataFrame, config: dict[str, int] | None = None) -> pd.DataFrame:
    """一次性算出策略层需要的全部特征列，返回新表（不修改入参）。"""
    cfg = {"atr": 14, "fast": 20, "slow": 50, "donchian": 20, "exit": 10, "adx": 14, "boll": 20}
    cfg.update(config or {})
    out = frame.copy()
    close, high, low = out["close"], out["high"], out["low"]
    out["atr"] = atr(high, low, close, cfg["atr"])
    out["atr_pct"] = out["atr"] / close
    out["ema_fast"] = ema(close, cfg["fast"])
    out["ema_slow"] = ema(close, cfg["slow"])
    out["donchian_up"], out["donchian_dn"] = donchian(high, low, cfg["donchian"])
    out["exit_up"], out["exit_dn"] = donchian(high, low, cfg["exit"])
    out["adx"] = adx(high, low, close, cfg["adx"])
    out["er"] = efficiency_ratio(close, cfg["fast"])
    out["boll_z"] = bollinger_z(close, cfg["boll"])
    out["rsi"] = rsi(close, cfg["atr"])
    out["vol_annual"] = realized_vol(close, 20)
    out["vol_pctile"] = percentile_rank(out["vol_annual"], 244)
    out["slope"] = slope(close, cfg["slow"])
    return out


def describe_snapshot(features: pd.DataFrame) -> dict[str, float | str]:
    """取最新一根 bar 的特征值，给操作清单用。"""
    last = features.dropna(subset=["atr"]).iloc[-1]
    regime = "趋势" if float(last["adx"]) >= 25 else ("震荡" if float(last["adx"]) < 20 else "过渡")
    return {
        "date": str(pd.to_datetime(last["date"]).date()),
        "close": float(last["close"]),
        "atr": float(last["atr"]),
        "atr_pct": float(last["atr_pct"]),
        "adx": float(last["adx"]),
        "er": float(last["er"]),
        "boll_z": float(last["boll_z"]),
        "rsi": float(last["rsi"]),
        "vol_annual": float(last["vol_annual"]),
        "vol_pctile": float(last["vol_pctile"]) if np.isfinite(last["vol_pctile"]) else float("nan"),
        "slope": float(last["slope"]),
        "regime": regime,
        "donchian_up": float(last["donchian_up"]) if np.isfinite(last["donchian_up"]) else float("nan"),
        "donchian_dn": float(last["donchian_dn"]) if np.isfinite(last["donchian_dn"]) else float("nan"),
    }
