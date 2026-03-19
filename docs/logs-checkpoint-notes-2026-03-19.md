# Logs 目录与检查点保存说明（2026-03-19）

## 现象
训练过程中或中止后，logs 目录经常看起来没有变化，容易误判为“该目录无用途”。

## 根因
在原训练逻辑中，模型保存依赖于严格条件：
- 仅当 `eval_score > best_val_score` 才会保存。
- 如果首轮评估值与初始 best 值相同（例如都为 0.0），则不会触发保存。
- 用户手动中断训练（KeyboardInterrupt）时，没有中断检查点保存逻辑。

因此可能出现：
- 训练跑了一段时间，但 logs 内没有新文件。
- 手动中止后，进度无法恢复。

## 本次修复
已在 [main_arcface.py](main_arcface.py) 调整检查点策略：

1. 每轮保存最新检查点
- 每个 epoch 结束后都会保存：`<name>.latest`
- 作用：保证训练随时可恢复，不依赖“是否达到最优”。

2. 最优检查点独立保存
- 当 `eval_score >= best_val_score` 时保存：`<name>`
- 作用：保留当前最佳模型。

3. 中断兜底保存
- 捕获 KeyboardInterrupt 时保存：`<name>.interrupt`
- 作用：用户停止训练后仍可从中断点恢复。

4. 保存目录自动创建
- 若目标目录不存在，自动创建后再保存。

## 修复后的预期文件
在 `--name test-VQA` 时，logs 目录下可能出现：
- `test-VQA`（best）
- `test-VQA.latest`（latest）
- `test-VQA.interrupt`（仅手动中断时出现）

## 结论
logs 目录不是文本日志目录，而是模型检查点目录。修复后它会在训练过程中持续更新，具备明确用途。
