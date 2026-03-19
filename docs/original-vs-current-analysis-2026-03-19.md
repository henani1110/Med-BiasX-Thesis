# 原始基线与当前版本差异分析（2026-03-19）

## 1. 分析范围与基线定义
- 本文将“最原始代码”定义为提交：`08598b0`。
- 当前分析目标版本为：`HEAD`（当前分支最新提交）。
- 差异命令来源：`git diff 08598b0..HEAD`。

说明：`08598b0` 是本轮连续修复与设计改造前的基线点，适合作为你后续论文实验与新对话上下文的参考锚点。

## 2. 总体变更规模
- 变更文件数：14
- 新增行数：1698
- 删除行数：94

## 3. 文件级差异总览

### 3.1 核心代码（功能行为变化）
- [main_arcface.py](main_arcface.py)
- [train_arcface.py](train_arcface.py)
- [modules/base_model_arcface.py](modules/base_model_arcface.py)
- [modules/base_model_arcface_qtype.py](modules/base_model_arcface_qtype.py)
- [utils/utils.py](utils/utils.py)

### 3.2 文档（分析与变更记录）
- [docs/change-summary-2026-03-19.md](docs/change-summary-2026-03-19.md)
- [docs/logs-checkpoint-notes-2026-03-19.md](docs/logs-checkpoint-notes-2026-03-19.md)
- [docs/nan-stability-first-principles-2026-03-19.md](docs/nan-stability-first-principles-2026-03-19.md)

### 3.3 备份（对照实验与回溯）
- [backups/pre-nan-fix-2026-03-19/base_model_arcface.py](backups/pre-nan-fix-2026-03-19/base_model_arcface.py)
- [backups/pre-nan-fix-2026-03-19/base_model_arcface_qtype.py](backups/pre-nan-fix-2026-03-19/base_model_arcface_qtype.py)
- [backups/pre-nan-fix-2026-03-19/train_arcface.py](backups/pre-nan-fix-2026-03-19/train_arcface.py)
- [backups/pre-warmup-post-nan-2026-03-19/main_arcface.py](backups/pre-warmup-post-nan-2026-03-19/main_arcface.py)
- [backups/pre-warmup-post-nan-2026-03-19/train_arcface.py](backups/pre-warmup-post-nan-2026-03-19/train_arcface.py)
- [backups/pre-warmup-post-nan-2026-03-19/README.md](backups/pre-warmup-post-nan-2026-03-19/README.md)

## 4. 与基线相比的关键行为变化

### 4.1 训练入口与运行流程（main_arcface.py）
1. 新增参数：`--base-model`、`--feat-dim`。
- 原因：数据集构造器依赖 `args.base_model`，基线代码传入参数不足会触发属性错误。

2. 新增参数：`--aux-warmup-epochs`、`--aux-loss-weight`。
- 原因：为 SupCon/DDC/ECC 提供 warmup 机制，缓解训练早期梯度冲击。

3. 设备策略从固定 CUDA 改为自动设备选择。
- 当前：自动选择 `cuda` / `mps` / `cpu`。
- 影响：macOS（无 CUDA）可直接训练，不再在 `.cuda()` 处崩溃。

4. run 目录与报告机制。
- 每次新训练创建独立目录：`logs/<name>-<timestamp>/`。
- 自动生成并持续刷新 `run-report.md`。
- 影响：实验可追踪性提升，便于多次对照。

5. 检查点策略增强。
- 保存 `best`、`latest`、`interrupt`。
- 影响：支持中断恢复与结果复盘，降低训练中断损失。

### 4.2 损失与优化稳定性（train_arcface.py）
1. SupCon 公式改为稳定实现。
- 引入 mask、对角剔除、分母下限保护和 `nan_to_num`。
- 影响：减少 `exp/log` 引起的数值异常。

2. DDC 稳定化。
- KL 使用 `batchmean`。
- 比值分母加 `EPS`。
- 使用 `log1p`、指数输入裁剪、loss 上限裁剪。
- 影响：避免 `kl1/kl2` 和 `exp(kl1-kl2)` 爆炸。

3. ECC 链路保护。
- 对 `a_logits`、`ce_logits`、`q_logits` 限幅后再参与 DDC/ECC。
- 影响：减小极端 logits 引发的梯度尖峰。

4. 梯度与更新安全策略。
- 梯度裁剪覆盖 `model + m_model` 两个分支。
- 检测非有限梯度并跳过异常 step。
- 影响：防止一次坏梯度污染参数并导致全局 NaN。

5. warmup 接入辅助损失。
- SupCon、DDC、ECC 使用统一 `aux_scale` 线性升权。
- 影响：训练前期更稳，减少无 warmup 下梯度爆炸概率。

6. 设备兼容更新。
- 训练/评估中的张量迁移改为 `.to(DEVICE)`。

### 4.3 ArcMargin 数值修复（base_model_arcface*.py）
1. `learned_mg` 去除 `.double()`。
- 原因：MPS 不支持 float64，基线实现会在 Apple Silicon 上报错。

2. `cosine` 限幅到接近 [-1, 1]。
- 原因：防止 `sqrt(1-cos^2)` 在边界数值误差时出现 NaN。

### 4.4 工具函数兼容性（utils/utils.py）
1. 移除 `torch._six` 依赖，改用 `string_classes = (str, bytes)`。
2. `collections.Mapping/Sequence` 改为 `collections.abc`。
- 影响：兼容新版本 PyTorch 与 Python 3.10+。

## 5. 与论文复现目标相关的现状判断
1. 已解决的核心阻塞
- NaN 与无 CUDA/MPS 兼容问题。
- 训练流程可持续运行并能保存有效检查点与报告。

2. 尚未完全对齐论文协议的部分
- 当前训练中使用 test 频繁验证，存在评估泄漏风险。
- 需要切换为 val 选模、test 最终一次报告。
- 需要输出与论文表格一致的 All/Open/Closed 口径汇总。

## 6. 对照实验资产清单（建议长期保留）
1. NaN 修复前：
- [backups/pre-nan-fix-2026-03-19/train_arcface.py](backups/pre-nan-fix-2026-03-19/train_arcface.py)
- [backups/pre-nan-fix-2026-03-19/base_model_arcface.py](backups/pre-nan-fix-2026-03-19/base_model_arcface.py)
- [backups/pre-nan-fix-2026-03-19/base_model_arcface_qtype.py](backups/pre-nan-fix-2026-03-19/base_model_arcface_qtype.py)

2. NaN 修复后、Warmup 前：
- [backups/pre-warmup-post-nan-2026-03-19/main_arcface.py](backups/pre-warmup-post-nan-2026-03-19/main_arcface.py)
- [backups/pre-warmup-post-nan-2026-03-19/train_arcface.py](backups/pre-warmup-post-nan-2026-03-19/train_arcface.py)

3. 当前版本（NaN 修复 + Warmup + 报告机制）：
- [main_arcface.py](main_arcface.py)
- [train_arcface.py](train_arcface.py)

## 7. 新对话续接建议（避免上下文丢失）
建议在新对话第一条消息直接提供以下信息：
1. 基线提交：`08598b0`。
2. 当前目标：论文口径复现（SLAKE-BIAS / VQA-RAD-BIAS 的 All/Open/Closed）。
3. 当前采用版本：NaN 修复 + warmup。
4. 对照资源位置：
- [backups/pre-nan-fix-2026-03-19](backups/pre-nan-fix-2026-03-19)
- [backups/pre-warmup-post-nan-2026-03-19](backups/pre-warmup-post-nan-2026-03-19)
5. 分析文档入口：
- [docs/original-vs-current-analysis-2026-03-19.md](docs/original-vs-current-analysis-2026-03-19.md)
- [docs/nan-stability-first-principles-2026-03-19.md](docs/nan-stability-first-principles-2026-03-19.md)
