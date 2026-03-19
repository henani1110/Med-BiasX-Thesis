# Med-BiasX 修改说明（2026-03-19）

## 背景
本次修改目标是解决以下问题：
1. 新版 PyTorch 与旧代码不兼容导致启动失败（如 `torch._six`）。
2. macOS（Apple Silicon）环境无 CUDA 时训练脚本崩溃。
3. 运行链路中的参数传递、键类型与保存路径问题导致中途异常。

## 逐文件变更与原因

### 1. main_arcface.py
- 变更内容：
  - 新增参数：`--base-model`、`--feat-dim`。
  - 将 `VQAFeatureDataset(..., dataset)` 改为 `VQAFeatureDataset(..., args)`。
  - 增加设备选择逻辑：自动使用 `cuda` / `mps` / `cpu`。
  - 将 `model.cuda()` / `metric_fc.cuda()` 改为 `.to(device)`。
  - 仅在 CUDA 可用时设置 `CUDA_VISIBLE_DEVICES`、`cudnn` 与 `torch.cuda.manual_seed`。
  - 在保存模型前自动创建保存目录。
- 修改原因：
  - 数据集构造函数依赖 `args.base_model` 等字段，原先传字符串会报 `AttributeError`。
  - macOS 无 CUDA，硬编码 `.cuda()` 会直接失败。
  - 保存到 `logs/...` 时目录不存在会抛出 `RuntimeError`。

### 2. utils/utils.py
- 变更内容：
  - 移除 `from torch._six import string_classes`。
  - 增加 `string_classes = (str, bytes)`。
  - 使用 `from collections.abc import Mapping, Sequence`，并替换旧式 `collections.Mapping/Sequence` 判断。
- 修改原因：
  - `torch._six` 在新版本 PyTorch 中已移除，导入会报 `ModuleNotFoundError`。
  - Python 3.10+ 中推荐使用 `collections.abc`，提高兼容性。

### 3. train_arcface.py
- 变更内容：
  - 增加全局 `DEVICE` 自动选择逻辑（`cuda`/`mps`/`cpu`）。
  - 将训练与评估流程中的 `.cuda()` 全部替换为 `.to(DEVICE)`。
  - 修复 `compute_supcon_loss`：
    - 对张量标签与非张量标签分别处理。
    - 去除会触发作用域错误的写法，改为稳健映射逻辑。
  - 评估统计键统一转成字符串，避免以张量作为 dict key 产生 `KeyError`。
  - 评估打印时使用 `str(key)`，避免字符串拼接类型错误。
- 修改原因：
  - 原代码假设 CUDA 必然可用，导致 macOS 运行失败。
  - 监督对比损失中标签映射在当前环境触发 `NameError`。
  - 评估阶段出现 `TypeError` 与 `KeyError`，需要统一 key 类型。

### 4. modules/base_model_arcface.py
- 变更内容：
  - 将 `learned_mg.double()` 相关逻辑改为保持 `float32`：
    - `torch.where(m > 1e-12, learned_mg, torch.full_like(learned_mg, -1000.0)).float()`
- 修改原因：
  - MPS 不支持 float64，原先 `.double()` 会报 dtype 不支持错误。

### 5. modules/base_model_arcface_qtype.py
- 变更内容：
  - 与 `base_model_arcface.py` 相同，去除 `.double()`，保持 `float32` 计算路径。
- 修改原因：
  - 同样为解决 MPS 下 float64 不支持问题，避免运行中断。

## 运行结果
- 启动、数据加载、训练与评估流程可以在 `medbiasx` 环境下继续执行。
- 1 个 epoch 的 smoke test 可运行完成，并可进入保存逻辑。

## 备注
- 本次提交不包含运行期自动生成文件（如 `__pycache__/`、`runs/`），仅包含源码修复与本说明文档。
