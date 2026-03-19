# Logs 目录与训练记录说明（2026-03-19）

## 设计目标
为每次训练生成独立目录，避免不同实验互相覆盖，并提供可直接阅读的结果文档。

## 当前实现
已在 [main_arcface.py](main_arcface.py) 实现以下机制：

1. 每次新训练创建独立 run 目录
- 示例：`logs/test-VQA-20260319-173200/`
- 目录内仅存放这一次训练的检查点和报告。

2. 检查点分角色保存
- `test-VQA`：best 检查点。
- `test-VQA.latest`：每个 epoch 都刷新。
- `test-VQA.interrupt`：仅在手动中止训练时保存。

3. 自动生成 run 报告文档
- 每个 run 目录都会生成 `run-report.md`。
- 报告包含：
	- 运行状态（running / finished / interrupted）
	- 数据集、模型、设备、目标 epoch
	- latest_epoch、best_epoch、best_val_score
	- best/latest/interrupt 文件是否存在、路径、大小、更新时间
	- 本次运行使用的完整参数（Args）

4. 报告会在训练过程中持续刷新
- 每个 epoch 结束后更新一次 `run-report.md`。
- 训练结束或中断时再更新一次最终状态。

## 使用方式
命令示例：

```bash
python main_arcface.py --name test-VQA --gpu 0 --dataset slake-cp
```

运行后查看：
1. `logs/` 下新建的本次 run 目录。
2. run 目录中的 `run-report.md`。

## 说明
- logs 目录定位为“模型检查点与实验报告目录”，不是纯文本训练日志目录。
- 这种结构同时满足：
	- 快速定位单次实验结果。
	- 明确区分 best 与 latest。
	- 训练中断时可追溯与恢复。
