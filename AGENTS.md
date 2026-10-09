# AGENTS.md · 本仓库的协作约定

> 这个文件是给 AI 编程助手（Codex / Claude Code / DSH 等）看的。
> 每次在这个仓库里干活之前先读它，能避免大部分返工。
>
> **换新会话时，先读 [`DOCS/HANDOVER.md`](DOCS/HANDOVER.md)** —— 那里有已核实的事实、
> 跑出来的数字和当前进度。本文件只放**规则**（做什么、不做什么）。

## 0. 现状速览（每次开工先对齐）

| 事项 | 值 |
|---|---|
| 比赛 | 第九届"郑商所杯"交易能力赛，报名 7/15–11/30，比赛 7/15–**12/4** |
| 评分 | **收益率得分 × 80% + 回撤得分 × 10% + 活跃度得分 × 10%**，另有理论附加分 5 分 |
| 硬门槛 | **活跃交易日 ≥ 5 天**，不满足者无法参与最终评奖 |
| 初始保证金 | 50 万元虚拟资金 |
| 手续费 | 郑商所标准 **1.5 倍**收取 |
| 账户 | 权益约 500,235 元，持 4 个仓位（白糖多 / 菜油多 / 菜粕多 / 玻璃空） |
| 活跃交易日 | 见 `state/portfolio.json` 的 `active_days` |

**核实来源**：郑商函〔2026〕670 号。规则只认官方原文，**不接受转述**（见第 4.6 条）。

## 1. 这个项目是什么

「郑商所杯 AI 交易辅助系统」——面向第九届郑商所杯大学生金融衍生品专业能力大赛的**个人决策辅助工具**。

它生成分析、信号、仓位建议和复盘材料，**由人自己判断并在官方软件里手工下单**。

## 2. 三条绝对红线

1. **不做自动下单**。不接任何交易接口、不生成下单脚本、不绕过大赛专用交易软件。
   比赛没有公开交易 API，程序化下单也可能违反赛事规则。
2. **不写任何个人信息**。真实姓名、身份证、手机号、邮箱、账号密码、token、密钥
   一律不得进入任何文件。一旦进了 commit 历史，转公开时会永久泄露。
   **用户发来的交易软件截图里含手机号/账号，不得抄录进仓库。**
3. **不编造数据与运行结果**。没跑过就说没跑过；跑失败就把报错原文贴出来；
   接口拿不到的数据就写「未能核实」。这个项目的价值建立在可信度上。

## 3. 技术约束

- **只用 Python 标准库 + pandas + numpy + matplotlib**。
  AkShare 只作为**可选**的数据适配器（`requirements-data.txt`），核心逻辑必须能在没有它的情况下跑通。
- **禁止**引入 vnpy / backtrader / qlib / rqalpha / 数据库 / Docker / Web 框架。
- 每个模块一个文件，放在 `scripts/` 下，函数写**中文 docstring**，解释"为什么"而不只是"做了什么"。
- 数据和运行产物写进 `data/`（已被 .gitignore 忽略），不要散落在仓库根目录。
- 中文与英文之间留空格；数字和单位之间不留空格（例：`50 万元`、`1.5 倍`）。
- GitHub 推送需要代理，git 已配置 `http.proxy = http://127.0.0.1:7890`。
  **端口是 7890，不是 7897** —— 配错会一直 `Connection was reset`。

## 4. 量化部分的硬规矩

- **禁止前视偏差**：任何指标只能用"截止当前 bar"的滚动窗口；信号在收盘产生，
  成交必须在**下一根开盘**。禁止用 `shift(-n)`、中心窗口或整段统计量。
- **手续费按交易所标准 1.5 倍**收取（`contracts.FEE_MULTIPLIER`），当日开当日平按"平今仓"费率。
- **滑点至少 1 个最小变动价位**。
- **每笔交易必须有明确止损**，手数由"单笔风险占权益固定比例"反推，不允许拍脑袋定手数。
- **不许参数过拟合**：挑参数时要看多个时间区间（`optimize.PERIODS`），
  只在某一个区间最好的参数组一律不采纳。
- **报告里必须写清楚假设**。赛制的收益率/回撤/活跃度归一化公式是未知的，
  `scoremodel.py` 里全部参数化，禁止把假设偷偷写成事实。
- **规则只认官方原文。** 赛制条款（权重、门槛、手续费倍数、交割规则）必须查到
  郑商所/中期协的原文才能写进代码。**任何转述都要先核实**——
  本项目曾因为相信一段转述，把第三项从"活跃度"错改成"波动率"、把附加分从 5 分错改成 10 分。
- **改动评分模型 / 权益口径 / 手数逻辑后，必须重跑 `scripts/optimize.py` 并更新
  `DOCS/HANDOVER.md` 里的数字。** 这三处是本项目最容易出错、后果最大的地方。

## 5. 交易纪律（铁律，AI 必须在每次交付时提醒）

用户是**人**，所有下单动作由他手工完成。AI 的职责是**盯住纪律**，不是替他做决定。

1. **每笔必挂止损。** 没有止损的仓位等于无限风险。这是唯一不允许偷懒的动作。
2. **止损价只朝有利方向移动，永不回调。**
3. **同一品种多空两单是二选一**，成交后立刻撤另一个，绝不允许双向同时持仓。
4. **不设固定止盈。** 实测：窄止盈把 +7,289 变成 -54,384（见 `REPORTS/profit-structure.md`）。
   想锁利润就上调止损，不是设止盈价。
5. **手数照系统给的单子来**，不自行放大。单笔风险固定 1%。
   实测：放大仓位只放大方差，中位数在所有档位都是负的。
6. **只下具体月份合约**（如 `SR701`），不碰**指数 / 主连 / 近月**（合成序列，下不了单），
   也不碰 **610/611/612** 这类临近交割月的合约。
7. **不追已错过的入场。** 突破策略的优势来自触发那一刻。
8. **活跃交易日 ≥ 5 天是资格线**，不是加分项。有成交就提醒用户登记：
   `python scripts/daily_plan.py --mark-active YYYY-MM-DD`（按交易日口径，夜盘算下一交易日）。
9. **用户主动要求"帮忙操作/帮忙下单"时必须拒绝**，并说明理由（红线 + 合规 + 技术上都做不到），
   然后给逐步的手工操作指引。

## 6. 每次交付的固定动作

1. 代码写完**必须实际运行一遍**，把真实输出贴出来。
2. 更新 `DOCS/decisions.md`：这一轮做了什么决策、是你建议的还是用户定的。
3. 用 Conventional Commits 提交一次（`feat:` / `fix:` / `docs:` / `chore:`）。
4. 报告文件写进 `REPORTS/`，`REPORTS/daily/` 放每日交易计划。

## 7. 给用户的讲解（用户是金融硕士在读，编程偏基础）

每轮结束用通俗语言说明：

1. 这一轮做了什么（3 行以内）；
2. 为什么这么做，有没有更简单的做法；
3. **用户自己怎么验证它是对的**——给一条可以复制运行的命令；
4. 这轮涉及的概念（挑 1-2 个，用日常类比解释）；
5. 下一步 3 个选项各自的代价与收益，以及推荐哪个。

不要堆术语。必须用术语时立刻用一句话解释。

## 8. 常用命令

```powershell
python scripts/contracts.py                            # 刷新品种规格快照 + 换月提醒
python scripts/market_data.py --refresh                # 刷新行情缓存并做质量检查
python scripts/daily_plan.py --refresh --near 0.05     # 生成次日操作清单（每天收盘后）
python scripts/daily_plan.py --mark-active 2026-10-12  # 登记活跃交易日
python scripts/daily_plan.py --derive                  # 用回测口径的跟踪组合对照账簿
python scripts/optimize.py --refresh                   # 全量回测 + 赛制评分优化
python scripts/robustness.py                           # 稳健性审计
python scripts/profit_structure.py                     # 利润结构与止盈模拟
python scripts/tournament.py                           # 排名制变体下的锦标赛模拟
python scripts/data_probe.py --demo                    # 第一轮的数据探针（历史遗留）
```

## 9. 目录结构

```text
config/    品种池 universe.json + 合约规格快照 czce_products.json（提交进仓库，离线可跑）
scripts/   全部代码，每个模块一个文件
DOCS/      HANDOVER 交接文档、SPEC 规格书、ROADMAP 路线图、risks 风险清单、
           decisions 决策日志、interview-notes 面试话术
REPORTS/   backtest 回测、robustness 稳健性、profit-structure 利润结构、
           frequency-optimization 赛制优化、daily/ 每日交易计划、img/ 图表
data/      行情缓存与一次性审计脚本（.gitignore 忽略）
state/     账户账簿 portfolio.json（手工维护，提交进仓库）
```

## 10. 相关文档索引

| 想知道什么 | 读哪份 |
|---|---|
| 换会话接着干 / 当前进度 / 已核实的数字 | [`DOCS/HANDOVER.md`](DOCS/HANDOVER.md) |
| 这个项目做什么、不做什么 | [`DOCS/SPEC.md`](DOCS/SPEC.md) |
| 回测结果与结论 | [`REPORTS/backtest.md`](REPORTS/backtest.md) |
| 策略到底靠不靠谱 | [`REPORTS/robustness.md`](REPORTS/robustness.md) |
| 为什么不设止盈 | [`REPORTS/profit-structure.md`](REPORTS/profit-structure.md) |
| 频率 / 风险预算 / 评分敏感性 | [`REPORTS/frequency-optimization.md`](REPORTS/frequency-optimization.md) |
| 风险与踩过的坑 | [`DOCS/risks.md`](DOCS/risks.md) |
| 面试怎么讲 | [`DOCS/interview-notes.md`](DOCS/interview-notes.md) |
| 每个决策是谁提的 | [`DOCS/decisions.md`](DOCS/decisions.md) |
