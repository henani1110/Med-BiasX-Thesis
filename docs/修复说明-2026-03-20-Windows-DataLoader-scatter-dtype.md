# 修复说明（2026-03-20）：Windows DataLoader `scatter_` 索引类型报错

## 问题现象

在 Windows + CUDA 环境训练时，DataLoader worker 抛出错误：

- `RuntimeError: scatter(): Expected dtype int64 for index`

错误位置在 `utils/dataset.py` 的 `target.scatter_(0, labels, scores)` 及同类 `scatter_` 调用。

## 根因分析

`scatter_` 的 index 参数必须是 `torch.int64`（`torch.long`）。
在不同平台/依赖组合下，`labels`、`margin_label`、`freq_label` 可能来自 `numpy` 或其他类型，未必是 `int64`，从而在 Windows 上被严格校验触发异常。

## 修复内容

文件：`utils/dataset.py`

在执行 `scatter_` 前，显式进行类型规范化：

1. 索引统一转为 `torch.long`：
   - `labels`
   - `margin_label`
   - `freq_label`
2. 数值张量转为目标张量对应 dtype：
   - `scores` -> `target.dtype`
   - `margin_score` -> `target_margin.dtype`
   - `per0` -> `freq_margin0.dtype`

## 影响评估

1. **对 Windows**：修复 `scatter_` 的 dtype 报错，训练可继续。
2. **对 macOS/Linux**：行为保持一致，仅做类型显式规范化，无训练逻辑变化。
3. **性能影响**：仅增加少量张量类型转换，开销很小。

## 回归建议

1. Windows 上复跑此前报错命令，确认不再出现 `Expected dtype int64 for index`。
2. 在 macOS 上做 1 epoch smoke test，确认指标与流程正常。
