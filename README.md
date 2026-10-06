# 郑商所杯 AI 交易辅助系统

> 一个面向第九届“郑商所杯”大学生金融衍生品专业能力大赛的个人决策辅助项目。

## 项目愿景

本项目探索如何在收益率、回撤与活跃度共同计分，且手续费按交易所标准 1.5 倍收取的约束下，辅助个人参赛者形成可解释、可复盘的交易决策。项目定位是**决策辅助系统**：生成分析、信号、风控建议和复盘材料，由参赛者自行判断并在官方软件中手工操作。

项目不会自动下单，也不声称能够保证收益。赛制细节与第三方辅助工具规则仍需参赛者以赛事官方文件核验。

## 项目状态

已完成第一轮数据探针：抓取 30 分钟行情、保存本地 CSV、生成基础质量报告；支持离线检查和明确标注的模拟数据。尚未实现策略、回测、定时更新或交易功能。

2026-10-06 实测：AkShare 返回 CF0 示例的最近 300 根数据，覆盖 2026-08-25 13:45 至 2026-09-30 15:00。**基础检查通过不代表历史完整或数据实时，也未证明能获取 2–3 年分钟历史。**

## Project Vision

This repository explores a personal decision-support workflow for the 2026 Zhengzhou Commodity Exchange Cup. It focuses on balancing return, drawdown, trading activity, and the competition's 1.5x exchange-fee assumption.

The system is intended to provide analysis, signals, risk guidance, and review materials. The participant remains responsible for every decision and places orders manually through the official competition software. **This project does not place orders automatically and does not promise investment returns.** Competition rules and restrictions on third-party assistance must be verified against official materials.

## Status

A single-file data probe is available. It fetches intraday futures bars, stores local CSV files, and reports basic data quality. Offline CSV validation and clearly labelled simulated data are supported. Strategies, backtests, and scheduled updates are not implemented yet.

## Code / 代码

在仓库根目录运行。已验证环境：Python 3.12.7、pandas 2.2.1、numpy 1.26.4、AkShare 1.18.64。Python 3.13 尚未验证。

```powershell
# 推荐使用项目虚拟环境；离线检查和模拟模式只需基础依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 scripts/data_probe.py --demo

# 联网抓取需要额外安装 AkShare，仅用作数据适配器
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -X utf8 scripts/data_probe.py --symbol CF0 --period 30 --limit 300

# 离线重新检查已保存的真实样本
.\.venv\Scripts\python.exe -X utf8 scripts/data_probe.py --input data/probe/CF0_30m_real.csv
```

- 数据和运行报告保存在 `data/probe/`，不上传 GitHub。
- 真行情、模拟数据、离线输入分别使用 `_real`、`_demo`、`_local` 文件名，不相互覆盖。
- 联网抓取最多等待 30 秒；失败时输出明确标注的模拟数据及降级原因，不自动使用旧行情。
- 原始数据校验失败时返回非零退出码，只更新对应的诊断报告，不覆盖已保存的 CSV。
- 检查字段、时间解析、重复、顺序、有限数值、价格区间及成交量；完整率和交易时段检查尚未实现。
- `CF0` 是接口连续合约示例，并非已选策略品种或可直接交易的具体合约。模拟序列包含非交易时段，只用于流程验证。

详见 [本轮报告](REPORTS/data-probe.md) 和 [决策日志](DOCS/decisions.md)。

## License

TODO: Choose a license before publishing source code for reuse.
