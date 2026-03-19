# Pre-Warmup Backup (Post-NaN Fix)

这个目录用于对比 warmup 的效果，特点如下：

- 已包含 NaN 稳定性修复（EPS/clamp/log1p/梯度有限性保护等）。
- 不包含 warmup 机制。

文件说明：
- main_arcface.py: 无 `--aux-warmup-epochs`、`--aux-loss-weight` 参数。
- train_arcface.py: 辅助损失项（SupCon/DDC/ECC）不做 warmup 缩放，直接全权重参与训练。

建议对比方式：
1. 使用本目录代码作为“无 warmup”组。
2. 使用项目当前代码作为“有 warmup”组。
3. 保持数据、seed、batch size、学习率一致，仅比较 warmup 开关影响。
