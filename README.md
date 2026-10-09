# 郑商所杯 AI 交易辅助系统

> 面向第九届「郑商所杯」大学生金融衍生品专业能力大赛的个人决策辅助工具。
> 生成信号、仓位与风控建议，**由人自己判断并在官方软件里手工下单**。
>
> 📌 **换会话接手 / 想知道现状**：先读 [`DOCS/HANDOVER.md`](DOCS/HANDOVER.md)
> —— 那里有已核实的事实、跑出来的数字、踩过的坑和当前进度。
> [`AGENTS.md`](AGENTS.md) 是给 AI 助手看的**规则**（红线 + 交易纪律）。
>
> 📋 **评分规则以官方原文为准**（郑商函〔2026〕670 号）：
> 收益率得分 × 80% + 回撤得分 × 10% + 活跃度得分 × 10%，
> 另有理论附加分 5 分；**活跃交易日 ≥ 5 天是资格门槛**，不满足者无法参与评奖。

## 项目定位

赛制把总分定成 **收益率 80% + 回撤 10% + 活跃度 10%**，手续费按交易所标准 **1.5 倍**收取，
而且必须用大赛专用软件**手工下单**（没有公开交易 API）。

这三个约束凑在一起会产生一个真实的矛盾：

- 想要活跃度分 → 得多交易；
- 1.5 倍手续费 + 滑点 → 多交易会吃掉占 80% 权重的收益。

本项目的价值就在这条矛盾上：**把「最优交易节奏」量化出来，并变成每个交易日一张照着就能挂单的操作清单。**

## 现在能做什么

一条命令生成下一个交易日的操作清单：

```powershell
python scripts/daily_plan.py --refresh
```

输出 `REPORTS/daily/YYYY-MM-DD.md`，包含：

- 每个品种的方向、**触发价**、几手、**初始止损**、单笔风险、预估手续费；
- 在手持仓**必须挂出的移动止损价**；
- 执行时刻表（含夜盘 21:00 / 日盘 09:00 的具体动作）；
- 组合风控状态（权益、回撤、剩余保证金额度）；
- **故意留空的复盘模板** —— 面试官最关心"哪些判断是你做的"。

## 核心结论（都是真实回测跑出来的，不是设想）

| 结论 | 证据 |
|---|---|
| 海龟法则的默认参数在本数据上最差 | 跟踪止损 3ATR 时全历史 +52.5%，放宽到 6ATR 后 +80.8% |
| **策略 edge 在 2016 年后大幅衰减** | 最好的参数组近 3 年仍为 -3.2%，近 1 年才转正 +3.3% |
| 为活跃度多交易不划算 | 活跃度只占 10% 权重，成本却由 80% 权重的收益承担 |
| 单笔风险 1% 在 40 天赛程里偏低 | 50 万 × 1% = 每笔只赌 5000 元，几乎拉不开差距 |
| 平今仓手续费差异是结构性的机会 | 棉花/白糖/PTA/菜粕/硅铁/锰硅**平今近乎免收**，苹果是开仓的 **2 倍** |

详见 [回测报告](REPORTS/backtest.md) 与 [赛制优化分析](REPORTS/frequency-optimization.md)。

## 项目状态

| 模块 | 状态 |
|---|---|
| 合约规格与品种池（26 个郑商所品种） | ✅ |
| 行情层：2005 年至今日线 + 缓存 + 脏数据清洗 | ✅ |
| 指标层：ATR/ADX/唐奇安/布林/效率比（全向量化、无前视） | ✅ |
| 策略状态机：收盘决策 → 次日开盘成交 → 盘中挂止损 | ✅ |
| 风控层：单笔风险定额 + 保证金上限 + 回撤守门 | ✅ |
| 组合回测：1.5 倍手续费 + 滑点 + 保证金约束 | ✅ |
| 赛制评分模型 + 频率/风险前沿 | ✅ |
| **每日交易计划生成器** | ✅ |
| 30 分钟日内补充模块 | ⬜ 未做（Should 档） |
| 结果 HTML 看板 | ⬜ 未做（Should 档） |

## 快速开始

```powershell
# 基础依赖（离线校验与回测只需这两个）
python -m pip install -r requirements.txt

# 联网抓行情需要额外装 AkShare
python -m pip install -r requirements-data.txt

# 1) 刷新郑商所品种规格快照 + 换月提醒
python scripts/contracts.py

# 2) 刷新行情缓存并做质量检查
python scripts/market_data.py --refresh

# 3) 全量回测 + 赛制优化，写 REPORTS/
python scripts/optimize.py --refresh

# 4) 生成下一个交易日的操作清单（每天收盘后跑一次）
python scripts/daily_plan.py --refresh

# 从空仓开始、想看系统跟踪组合的持仓，加 --derive
python scripts/daily_plan.py --derive --suffix -tracked
```

已验证环境：**Python 3.12.7、pandas 2.2.1、numpy 1.26.4、AkShare 1.18.64**（Python 3.13 未验证）。

## 目录结构

```text
config/    品种池 universe.json 与合约规格快照 czce_products.json（提交进仓库，离线可用）
scripts/   全部代码，每个模块一个文件
DOCS/      SPEC 规格书、ROADMAP 路线图、risks 风险清单、decisions 决策日志、interview-notes 面试话术
REPORTS/   backtest 回测报告、frequency-optimization 赛制优化、daily/ 每日交易计划
data/      行情缓存（.gitignore 忽略）
state/     账户账簿 portfolio.json（每次实际成交后手工更新）
```

## 模块一览

| 文件 | 职责 |
|---|---|
| `scripts/contracts.py` | 合约乘数、手续费、保证金、主力合约与换月提醒 |
| `scripts/market_data.py` | 日线/分钟线获取、缓存、脏数据清洗与质量报告 |
| `scripts/indicators.py` | 全部技术指标（滚动窗口，无前视偏差） |
| `scripts/strategy.py` | 唐奇安突破 + 布林均值回归状态机，输出成交事件与挂单价位 |
| `scripts/risk.py` | 手数反推、保证金约束、回撤守门 |
| `scripts/backtest.py` | 多品种组合回测（1.5 倍手续费 + 滑点） |
| `scripts/scoremodel.py` | 赛制评分模型与假设敏感性分析 |
| `scripts/optimize.py` | 参数扫描、频率扫描、风险预算前沿、报告生成 |
| `scripts/daily_plan.py` | **每日交易计划生成器（核心交付物）** |

## 已知局限

- **不做自动下单**，不接任何交易接口；所有输出都是给人看的清单。
- 新浪连续合约在**主力换月处有价格跳空**，是回测噪声来源；质量报告会统计并列出。
- 赛制的收益率/回撤/活跃度**归一化公式未知**，`scoremodel.py` 里全部参数化，
  结论用 5 组假设做稳健性检验，不把猜测写成事实。
- 所有结果都是**样本内**的，不是样本外验证。
- 本项目**不构成投资建议**，只服务比赛用虚拟账户。

详见 [风险清单](DOCS/risks.md) 与 [面试话术](DOCS/interview-notes.md)。

---

## Project Vision (English)

A personal decision-support system for the 2026 Zhengzhou Commodity Exchange Cup.
The competition scores **return × 80% + drawdown × 10% + activity × 10%**, charges
**1.5× exchange fees**, and offers **no public trading API** — orders must be placed
manually in the official terminal.

The system quantifies the resulting tension (more trades boost the activity score but
cost fees against the 80%-weighted return) and turns it into a daily, human-executable
order checklist.

**It does not place orders, does not connect to any trading API, and does not promise returns.**

```powershell
python scripts/optimize.py --refresh      # backtest + competition-score optimization
python scripts/daily_plan.py --refresh    # next trading day's executable plan
```

## License

TODO: Choose a license before publishing source code for reuse.
