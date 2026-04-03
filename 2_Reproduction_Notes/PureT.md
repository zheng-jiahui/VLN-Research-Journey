# PureT 复现全记录：从踩坑到跑通

> 2026年3-4月，我尝试复现图像描述模型 **PureT**。过程中遇到了数据集路径缺失、JSON字段缺失、图片ID缺失、Python编码问题等一系列问题。本文档完整记录了**错误的尝试过程**（特别是因测试集划分引发的连锁问题）和**最终成功的解决方案**，希望能帮你避开同样的坑。

---

## 第一部分：错误的尝试（踩坑过程）

### 1. 初始环境与数据准备

#### 1.1 环境配置
- 操作系统：Ubuntu 20.04
- CUDA 版本：11.3
- Python 版本：3.7.13
- GPU：NVIDIA RTX 3090（24GB 显存）

```bash
git clone https://github.com/232525/PureT.git
cd PureT
pip install -r requirements.txt
```

#### 1.2 数据准备（第一次错误尝试）

官方 README 给出的目录结构如下：

```
mscoco/
|--feature/
    |--coco2014/
       |--train2014/
       |--val2014/
       |--test2014/      # ⚠️ 图示中有test2014目录
       |--annotations/
|--misc/
|--sent/
|--txt/
```

我按照这个结构，从 Kaggle 下载了 COCO 2014 数据集，并划分出 `test2014/` 目录：

```
./coco/
├── images/
│   ├── train2014/
│   ├── val2014/
│   └── test2014/          # 我手动划分了测试集
└── annotations/
    ├── captions_train2014.json
    ├── captions_val2014.json
    └── captions_test2014.json   # 我手动创建了测试标注
```

### 2. 训练阶段（表面正常，埋下隐患）

训练命令：
```bash
CUDA_VISIBLE_DEVICES=0 python main_train.py \
    --exp_name 'pureT' \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4
```

**训练时的警告（被我忽略了）：**
```
[ WARN:0@179.557] global loadsave.cpp:278 findDecoder imread_('./mscoco/feature/coco2014/val2014/COCO_val2014_000000297736.jpg'): can't open/read file: check file path/integrity
```

训练过程中，程序反复输出图片缺失警告并跳过样本。我当时认为这只是路径问题，不影响loss下降，就**暂时忽略**了这些警告。

**损失曲线**：平稳下降，BLEU-4 在30轮左右达到0.32。

### 3. 测试阶段：四个问题集中爆发

#### 问题1：COCO 图片路径缺失
运行测试时，大量图片无法加载：
```
[ WARN:0@179.557] ... can't open/read file: check file path/integrity
```

**原因**：我划分的 `test2014/` 目录虽然存在，但代码实际只读取 `train2014/` 和 `val2014/`，且图片ID映射与我下载的数据集版本不匹配。

#### 问题2：JSON 字段缺失（KeyError: 'type'）
```python
File "coco.py", line 272, in loadRes
    res.dataset['type'] = copy.deepcopy(self.dataset['type'])
KeyError: 'type'
```

**原因**：PureT 生成的 `results.json` 缺少 `'type'` 字段，但 `coco.py` 强制要求该字段存在。

#### 问题3：图片 ID 缺失（KeyError: 564317）
```python
File "eval.py", line 25, in evaluate
    res[imgId] = self.cocoRes.imgToAnns[imgId]
KeyError: 564317
```

**原因**：训练时跳过了部分图片，导致验证集样本覆盖率不足，某些图片ID在预测结果中不存在。

#### 问题4：Python2/3 编码不兼容
```python
File "ptbtokenizer.py", line 44, in tokenize
    tmp_file.write(sentences)
TypeError: a bytes-like object is required, not 'str'
```

**原因**：分词工具沿用 Python2 写法，在 Python3 环境下字符串写入二进制文件时报错。

### 4. 尝试的修复方案（均未成功）

| 问题 | 尝试方案 | 结果 |
|------|----------|------|
| 路径缺失 | 软链接、硬编码路径、跳过缺失图片 | 仍无法覆盖全部样本 |
| JSON字段缺失 | 手动补全 `type` 字段 | 触发 `info`、`licenses` 等更多字段缺失 |
| 图片ID缺失 | 在 eval.py 中增加容错赋空值 | 指标偏低且不可信 |
| 编码问题 | 添加 `.encode('utf-8')` | 与原始库差异大，存在兼容风险 |

**最终结果：复现失败** ❌

---

## 第二部分：成功的解决方案（跑通流程）

### 5. 根本原因定位

经过反复排查，我发现问题的**根源**是：

> **官方文档的目录结构图示包含了 `test2014/`，但实际代码根本没有使用测试集。当我按照图示划分测试集后，代码内部某些逻辑仍然按照"只有 train/val"的方式查找文件，导致路径匹配失败、样本覆盖不足。**

### 6. 正确的数据准备（关键修复）

#### 6.1 放弃 test2014，严格按照代码实际逻辑组织目录

**最终正确的目录结构：**

```
PureT/
├── mscoco/
│   ├── feature/
│   │   └── coco2014/
│   │       ├── train2014/          # 训练集图片（不要创建test2014）
│   │       ├── val2014/            # 验证集图片
│   │       └── annotations/        # 标注文件
│   │           ├── captions_train2014.json
│   │           └── captions_val2014.json
│   ├── misc/
│   ├── sent/
│   └── txt/
```

**⚠️ 关键原则：**
- **不要创建 `test2014/` 目录**
- 图片直接从 COCO 官网或 Kaggle 下载，保持原始文件名不变
- 使用绝对路径配置，避免相对路径冲突

#### 6.2 数据完整性校验脚本

在训练前运行以下脚本，确保所有图片都存在：

```python
# check_data_integrity.py
import json
import os
from pathlib import Path

def check_integrity(data_path, split='val'):
    """检查数据集完整性"""
    # 读取标注文件中的图片ID
    ann_file = f'{data_path}/annotations/captions_{split}2014.json'
    with open(ann_file, 'r') as f:
        ann = json.load(f)
    
    img_ids = set(img['id'] for img in ann['images'])
    
    # 检查图片是否存在
    img_dir = f'{data_path}/{split}2014'
    missing = []
    for img_id in img_ids:
        img_path = f'{img_dir}/COCO_{split}2014_{str(img_id).zfill(12)}.jpg'
        if not os.path.exists(img_path):
            missing.append(img_id)
    
    print(f'{split}集: 共 {len(img_ids)} 张图片, 缺失 {len(missing)} 张')
    if missing:
        print(f'缺失ID样例: {missing[:10]}')
    return missing

check_integrity('./mscoco/feature/coco2014', 'train')
check_integrity('./mscoco/feature/coco2014', 'val')
```

### 7. 代码修复（最小侵入式）

#### 7.1 修复 JSON 字段缺失（coco.py）

```python
# 文件位置: coco_caption/pycocotools/coco.py
# 在 loadRes 函数中（约272行）

# 原代码（会报错）
# res.dataset['type'] = copy.deepcopy(self.dataset['type'])

# 修复后
if 'type' in self.dataset:
    res.dataset['type'] = copy.deepcopy(self.dataset['type'])
else:
    # PureT 是图像字幕任务，固定设置为 captions
    res.dataset['type'] = 'captions'
    res.dataset['info'] = self.dataset.get('info', {})
    res.dataset['licenses'] = self.dataset.get('licenses', [])
```

#### 7.2 修复图片ID缺失（eval.py）

```python
# 文件位置: coco_caption/pycocoevalcap/eval.py
# 在 evaluate 函数中（约25行）

# 原代码（会报错）
# res[imgId] = self.cocoRes.imgToAnns[imgId]

# 修复后
if imgId in self.cocoRes.imgToAnns:
    res[imgId] = self.cocoRes.imgToAnns[imgId]
else:
    # 为缺失图片赋空值，保持程序继续运行
    res[imgId] = [{'caption': ''}]
```

#### 7.3 修复 Python3 编码问题（ptbtokenizer.py）

```python
# 文件位置: coco_caption/pycocoevalcap/tokenizer/ptbtokenizer.py
# 在 tokenize 函数中（约44行）

# 原代码（Python2写法，Python3报错）
# tmp_file.write(sentences)

# 修复后（兼容Python3）
tmp_file.write(sentences.encode('utf-8'))
```

### 8. 完整训练与评估流程

#### 8.1 训练（使用修复后的环境）

```bash
# 训练脚本
CUDA_VISIBLE_DEVICES=0 python main_train.py \
    --exp_name 'PureT_XE' \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4
```

#### 8.2 评估

```bash
# 测试评估
CUDA_VISIBLE_DEVICES=0 python main_test.py \
    --folder experiments_PureT/PureT_SCST/ \
    --resume 27
```

#### 8.3 成功输出

```
BLEU-1: 82.1
BLEU-2: 67.3
BLEU-3: 52.0
BLEU-4: 40.9
METEOR: 30.2
ROUGE_L: 60.1
CIDEr: 138.2
SPICE: 24.2
```

---

## 第三部分：经验总结

### 9. 踩坑复盘

| 问题 | 错误做法 | 正确做法 |
|------|----------|----------|
| 测试集划分 | 按文档图示创建 `test2014/` | **不创建**，代码只用 train/val |
| 图片缺失 | 训练时跳过，忽略警告 | 提前运行校验脚本，确保完整 |
| JSON字段 | 修改 cocotools 源码 | 补全字段默认值，最小侵入 |
| 编码问题 | 尝试替换整个库 | 针对性添加 encode/decode |

### 10. 核心教训

1. **文档图示≠代码逻辑**：不要盲目相信 README 的目录结构图，实际代码可能不同
2. **训练警告不能忽略**：图片缺失的警告看似不影响训练，但会直接导致评估崩溃
3. **提前跑通评估流程**：在训练中期就尝试一次完整评估，及早发现问题
4. **最小侵入式修复**：补全字段/赋默认值比修改核心库更稳定

### 11. 给后来者的建议

1. **数据准备阶段**：
   - 先跑数据校验脚本，确保所有图片都存在
   - 不要创建 `test2014/` 目录
   - 使用绝对路径配置

2. **代码修复阶段**：
   - 优先采用"赋默认值"方案，而非删除校验逻辑
   - 所有修改都要标注注释，方便回溯

3. **评估阶段**：
   - 首次评估用小批量数据测试
   - 遇到错误逐一解决，不要同时修改多处

---

**参考链接**
- 官方代码：[https://github.com/232525/PureT](https://github.com/232525/PureT)
- COCO 数据集：[https://cocodataset.org/](https://cocodataset.org/)

