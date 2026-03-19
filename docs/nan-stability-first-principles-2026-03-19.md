# NaN 稳定性排查与修复说明（第一性原理）

## 目标
解决训练中 `loss=nan`，并在不破坏原始目标函数设计的前提下增强数值稳定性。

## 第一性原理排查路径
按照“前向有限性 -> 损失有限性 -> 梯度有限性 -> 参数更新有限性”分层排查：

1. 前向输出是否有限
- 检查 `ce_logits`、`q_logits`、`a_logits` 是否出现 `inf/nan`。

2. 损失子项是否有限
- SupCon：`exp/log` 与分母接近 0 的风险。
- DDC：`kl1/kl2`、`exp(kl1-kl2)`、`log(1 + ratio)` 风险。
- ECC：`logsumexp` 本身稳定，但上游 logits 过大仍会造成梯度冲击。

3. 梯度是否有限
- 即使 loss 有限，反向后也可能出现非有限梯度。

4. 更新是否污染参数
- 若一步更新包含非有限梯度，参数会被污染，后续 batch 全部崩溃。

## 与论文公式的对应关系（结合你提供的 Gemini 分析）
你给出的风险点与代码现象一致：
- DDC 中除法与指数项最危险；
- ECC 依赖 logits 质量，需控制输入范围；
- 单纯调小学习率通常不够，必须加数值保护和梯度保护。

## 已实施修复

### 1) SupCon 稳定化
文件： [train_arcface.py](train_arcface.py)
- 改为逐样本稳定形式，采用 masked log-prob；
- 分母 `clamp_min(EPS)`；
- 输出使用 `nan_to_num` 兜底。

### 2) DDC 稳定化
文件： [train_arcface.py](train_arcface.py)
- `KL` 使用 `reduction='batchmean'`；
- 分母加 `EPS`；
- `log(1+x)` 改为 `log1p(x)`；
- `exp` 输入做 `clamp`；
- `kl_loss` 做 `nan_to_num` + 上界裁剪。

### 3) ECC 链路保护
文件： [train_arcface.py](train_arcface.py)
- 在 DDC/ECC 前对 `a_logits/ce_logits/q_logits` 限幅到 `[-30, 30]`；
- `En` 做 `nan_to_num` + 上界裁剪。

### 4) ArcMargin 几何稳定化
文件：
- [modules/base_model_arcface.py](modules/base_model_arcface.py)
- [modules/base_model_arcface_qtype.py](modules/base_model_arcface_qtype.py)

改动：
- 对 `cosine` 限幅到接近 `[-1, 1]`，避免 `sqrt(1-cos^2)` 的边界数值误差。

### 5) 反向与更新保护
文件： [train_arcface.py](train_arcface.py)
- 梯度裁剪覆盖 `model + m_model` 两个分支；
- 检测非有限梯度，若异常则跳过该 step，避免参数污染。

### 6) 辅助损失 Warmup（你要求的第二项）
文件：
- [main_arcface.py](main_arcface.py)
- [train_arcface.py](train_arcface.py)

新增参数：
- `--aux-warmup-epochs`（默认 5）
- `--aux-loss-weight`（默认 1.0）

机制：
- SupCon、DDC、ECC 统一按线性系数升权：
  - $s = \min(1, \frac{epoch+1}{warmup}) \times w_{aux}$
- 训练初期减少辅助项冲击，后期恢复到目标权重。

## 你要的“未做 NaN 修复前”代码备份
已生成目录： [backups/pre-nan-fix-2026-03-19](backups/pre-nan-fix-2026-03-19)

包含三份文件：
1. [backups/pre-nan-fix-2026-03-19/train_arcface.py](backups/pre-nan-fix-2026-03-19/train_arcface.py)
2. [backups/pre-nan-fix-2026-03-19/base_model_arcface.py](backups/pre-nan-fix-2026-03-19/base_model_arcface.py)
3. [backups/pre-nan-fix-2026-03-19/base_model_arcface_qtype.py](backups/pre-nan-fix-2026-03-19/base_model_arcface_qtype.py)

这些备份从 `HEAD` 导出，代表本轮 NaN 修复前的版本。

## 你要的“Warmup 前（但已做 NaN 修复）”代码备份
已生成目录： [backups/pre-warmup-post-nan-2026-03-19](backups/pre-warmup-post-nan-2026-03-19)

包含文件：
1. [backups/pre-warmup-post-nan-2026-03-19/main_arcface.py](backups/pre-warmup-post-nan-2026-03-19/main_arcface.py)
2. [backups/pre-warmup-post-nan-2026-03-19/train_arcface.py](backups/pre-warmup-post-nan-2026-03-19/train_arcface.py)
3. [backups/pre-warmup-post-nan-2026-03-19/README.md](backups/pre-warmup-post-nan-2026-03-19/README.md)

这组备份用于“只比较 warmup 开关效果”：
- 保留 NaN 修复；
- 去除 warmup 参数与缩放逻辑。

## 使用建议
建议训练命令先采用：

```bash
python main_arcface.py --name stable-run --gpu 0 --dataset slake-cp --aux-warmup-epochs 5 --aux-loss-weight 1.0
```

若仍有不稳定，可尝试：
- 增大 `--aux-warmup-epochs` 到 8~12；
- 暂时降低 `--aux-loss-weight` 到 0.5 再逐步调回 1.0。
