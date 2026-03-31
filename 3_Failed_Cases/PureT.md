# PureT 复现踩坑实录：数据集缺失、JSON 报错、最终失败

> 2026年3月，我尝试复现一个图像描述（Image Captioning）模型 —— **PureT**。训练过程其实不算顺利。COCO 图片缺失 + JSON 字段缺失 + 评估编码问题 + 图片ID缺失，四个问题交织在一起，最终没能跑通评估流程。这篇文章就是这次失败的完整记录，希望能帮你在复现时避开同样的坑。

---

## 1. 初始问题与任务背景
### 1.1 复现的核心诉求（初始提问相关）
最初尝试复现 PureT 时，核心想解决两个问题：
1. 如何快速适配不同来源的 COCO 2014 数据集，解决图片路径缺失导致的训练警告问题；
2. 如何修复 pycocotools 评估时的 JSON 字段缺失（KeyError: 'type'）问题，让评估流程能正常跑完。

### 1.2 PureT 任务背景
**PureT** 是一个基于纯 Transformer 架构的端到端**图像描述（Image Captioning）**模型。简单来说，它的任务是：**输入一张图片，输出一句自然语言描述**（也就是经典的“图生文”任务）。

和 VLN（视觉-语言导航）不同，图像描述不需要智能体在环境中移动，只需要理解图像内容并生成对应的文本描述。PureT 的亮点在于：
- 完全基于 Transformer，摒弃了传统的 CNN+LSTM 结构；
- 在多个图像描述基准（如 COCO）上取得了不错的效果。

我选择复现它，主要是想学习 Transformer 在视觉-语言生成任务中的应用，为后续的图文生成在VLN领域的应用打基础，也希望能解决上述两个核心问题。

---

## 2. 环境配置

### 2.1 基础环境
- 操作系统：Ubuntu 20.04
- CUDA 版本：11.3
- Python 版本：3.7.13
- GPU：NVIDIA RTX 3090（24GB 显存）

### 2.2 依赖安装

官方代码仓库：  
[https://github.com/232525/PureT](https://github.com/232525/PureT)

```bash
git clone https://github.com/232525/PureT.git
cd PureT
pip install -r requirements.txt
```

依赖安装过程中没有遇到明显问题（PyTorch 版本用的是 1.8.0，与之前环境一致）。

---

## 3. 数据准备

### 3.1 COCO 数据集（第一个核心问题的起点）

PureT 使用 **COCO 2014** 数据集进行训练和评估。COCO 是图像描述任务最常用的数据集，包含：
- 训练集：约 8 万张图片，每张有 5 句人工标注的描述；
- 验证集：约 4 万张图片；
- 测试集：约 4 万张图片。

但是在README给的数据集连接失效，我只好在各种途径找数据集，最终我从 Kaggle 下载了 COCO 2014 数据集，解压后放在：

```
./coco/
├── images/
│   ├── train2014/
│   └── val2014/
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

**但问题存在**：
> Epoch 0 - train:   6%|▋          | 652/11328 [02:53<47:27,  3.75it/s, loss=5.47][ WARN:0@179.557] global loadsave.cpp:278 findDecoder imread_('./mscoco/feature/coco2014/val2014/COCO_val2014_000000297736.jpg'): can't open/read file: check file path/integrity

这正是我最初想解决的第一个问题：不同来源的 COCO 数据集文件匹配度不一致，导致模型训练时大量图片路径缺失，程序反复输出警告并跳过样本。无奈之下我先尝试跳过不存在的图片，但这也为后续评估埋下隐患。

### 3.2 图像特征提取
PureT 需要预先提取图像的网格特征（grid features）。官方提供了提取脚本，但依赖的 `bottom-up-attention` 库安装非常复杂。我尝试绕过后直接用原始图片，但发现代码中强制要求加载 `.tsv` 格式的特征文件。

---

## 4. 训练过程

### 4.1 训练命令

我按照官方提供的脚本进行训练：

```bash
CUDA_VISIBLE_DEVICES=0 python main_train.py \
    --exp_name 'pureT' \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --data_path ./coco \
    --features_path ./coco/features
```

### 4.2 训练观察
- **loss 曲线**：平稳下降，没有异常震荡。
- **验证集表现**：验证 loss 稳定下降，BLEU-4 指标在 30 轮左右达到 0.32（接近论文报告值）。
- **收敛时间**：约 35–40 个 epoch 后 loss 趋于平稳。

训练阶段看似正常，但我最初关注的两个核心问题还未暴露，直到测试阶段集中爆发。

---

## 5. 测试阶段 —— 四个核心问题集中爆发

### 5.1 第一个问题：COCO 图片路径缺失（初始提问核心问题1）

运行测试脚本时，控制台输出：

```
Epoch 0 - train:   6%|▋          | 652/11328 [02:53<47:27,  3.75it/s, loss=5.47][ WARN:0@179.557] global loadsave.cpp:278 findDecoder imread_('./mscoco/feature/coco2014/val2014/COCO_val2014_000000297736.jpg'): can't open/read file: check file path/integrity
```

**现象**：
- 程序没有直接崩溃，而是反复输出警告，跳过大量样本。
- 即使尝试修改路径映射、添加软链接，仍无法完全匹配原作者的数据集文件列表。

**临时处理**：
- 由于数据集我实在找不到和原作者一模一样，看着loss也在平稳下降，就暂时忽略这个警告，但这导致测试集样本覆盖率不足。

---

### 5.2 第二个问题：JSON 字段缺失（初始提问核心问题2）

在部分图片加载成功后，程序又抛出了新的错误：

```python
File "/home/d2r2/Matterport3DSimulator/PureT/coco_caption/pycocotools/coco.py", line 272, in loadRes
    res.dataset['type'] = copy.deepcopy(self.dataset['type'])
KeyError: 'type'
```

**现象**：
- 程序在调用 `coco.loadRes()` 时崩溃，无法继续评估。
- 堆栈指向 `coco.py` 的 `loadRes` 函数，它尝试从 `self.dataset` 中读取 `'type'` 字段，但该字段不存在。

**排查过程**：
1. 查看 `self.dataset` 的来源：它是通过加载 PureT 生成的 `results.json` 文件构建的。
2. 检查 `results.json` 的内容：发现只有 `"annotations"` 字段，缺少 `"type"`字段。
3. 对比标准 COCO 评估格式：标准格式应包含 `"type": "captions"`，但 PureT 的评估代码没有写入该字段。
4. 进一步溯源：在 `ICC分词预处理.ipynb` 中发现，`'type'` 是作者自定义添加的字段，并非 COCO 标准字段。

**解决尝试（针对初始提问的修复方案）**：
- 明确 PureT 是图像字幕任务（type:captions），直接修改 `coco.py` 中 `loadRes` 函数：
 ``` python
 # 原代码
 # res.dataset['type'] = copy.deepcopy(self.dataset['type'])
 # 修复缺失 type 字段的问题（针对核心问题2的修改）
if 'type' in self.dataset:
    res.dataset['type'] = copy.deepcopy(self.dataset['type'])
else:
    # PureT 是图像字幕任务，固定设置为 captions
    res.dataset['type'] = 'captions'
 ```

---

### 5.3 第三个问题：图片 ID 缺失导致 KeyError（新增问题1）

修复 `type` 字段后，程序继续运行，又触发了新的错误：

```python
File "/home/d2r2/Matterport3DSimulator/PureT/coco_caption/pycocoevalcap/eval.py", line 25, in evaluate
    res[imgId] = self.cocoRes.imgToAnns[imgId]
KeyError: 564317
```

**现象**：
- 评估代码在遍历验证集图片 ID 时，发现 ID `564317` 不存在于预测结果中，直接崩溃。
- 根源是之前训练时跳过了部分图片，导致验证集样本覆盖率不足，预测结果不完整。

**解决尝试**：
- 为了让程序继续运行，在 `eval.py` 中增加容错逻辑，对缺失 ID 赋空值：
 ```python
 # 原代码
 # res[imgId] = self.cocoRes.imgToAnns[imgId]
 # 修复缺失图片ID的问题
 if imgId in self.cocoRes.imgToAnns:
     res[imgId] = self.cocoRes.imgToAnns[imgId]
 else:
     res[imgId] = [{'caption': ''}]  # 标准空caption结构，避免拉低指标
 ```

---

### 5.4 第四个问题：Python2/3 编码不兼容（新增问题2）

解决图片 ID 缺失问题后，程序再次报错：

```python
File "/home/d2r2/Matterport3DSimulator/PureT/coco_caption/pycocoevalcap/tokenizer/ptbtokenizer.py", line 44, in tokenize
    tmp_file.write(sentences)
TypeError: a bytes-like object is required, not 'str'
```

**现象**：
- 分词工具 `ptbtokenizer.py` 沿用了 Python2 的写法，直接将字符串写入临时文件；
- 但在 Python3 中，临时文件默认以二进制模式打开，必须写入 bytes 类型，否则抛出类型错误。

**解决尝试**：
- 对写入语句添加编码转换，将字符串转为 UTF-8 字节流：
 ```python
 # 原代码
 # tmp_file.write(sentences)
 # 修复Python3编码问题
 tmp_file.write(sentences.encode('utf-8'))
 ```

---

## 6. 最终结果：复现失败

尽管针对初始提出的两个核心问题，以及后续新增的两个问题做了针对性处理，但仍未能通过测试阶段：
- 针对“路径缺失”：修改路径/软链接/跳过缺失图片，仍无法覆盖全部测试样本，评估结果不具备参考性；
- 针对“JSON 字段缺失”：手动补全 `type` 字段后，又触发了其他字段（如 `info`、`licenses`）的缺失问题，连锁报错无法终止；
- 针对“图片ID缺失”：添加容错逻辑后，虽然能继续运行，但部分样本被赋空值，导致评估指标偏低且不可信；
- 针对“编码问题”：修改编码后虽能正常分词，但代码与原始 pycocotools 库差异较大，存在潜在兼容性风险。

最终尝试的所有方案：
- 修正 COCO 图片路径，尝试软链接和硬编码；
- 手动修改 `coco.py`，添加缺失字段的默认值；
- 在 `eval.py` 中增加图片 ID 容错逻辑；
- 修复 `ptbtokenizer.py` 中的 Python3 编码问题；
- 替换为标准 pycocotools 并尝试适配；
- 在 GitHub Issues 中查找类似问题（未果）。

均未能解决根本问题，最终放弃了 PureT 的完整复现。训练了 50 个 epoch 的模型权重，最终无法在测试集上得到有效评估。

---

## 7. 失败原因复盘

### 7.1 直接原因
- **COCO 图片路径映射混乱**：代码中硬编码了多个相对路径，且与 Kaggle 下载的目录结构不匹配（对应初始核心问题1）。
- **评估 JSON 格式不完整**：PureT 对 pycocotools 的修改未完全适配，导致 `loadRes` 时字段缺失（对应初始核心问题2）。
- **样本覆盖率不足**：训练时跳过缺失图片，导致验证集预测结果不完整，触发图片 ID KeyError（对应 5.3 问题）。
- **Python 版本不兼容**：分词工具沿用 Python2 写法，在 Python3 环境下编码报错（对应 5.4 问题）。

### 7.2 根本原因
- **数据准备脚本不完整**：官方仓库没有提供一键式的数据准备脚本，用户需要自己理解 COCO 的目录结构要求。
- **评估模块与标准库耦合过紧**：对 pycocotools 的修改没有封装好，导致依赖内部字段结构，一旦生成 JSON 格式略有偏差就会崩溃。
- **社区支持不足**：该仓库的 Issues 区活跃度低，遇到问题时缺乏可参考的解决方案。
- **Python 版本适配缺失**：代码未做 Python3 兼容处理，在 Python3 环境下运行时触发编码问题。

---

## 8. 经验总结与建议（针对全部问题的补充）

### 8.1 对复现者的建议
- **针对路径缺失问题**：复现前先跑一遍数据校验脚本（如遍历数据集文件列表，对比代码中调用的路径），提前删除/标记缺失的图片，而非训练时临时跳过；优先使用绝对路径配置，避免代码中硬编码的相对路径冲突。
- **针对 JSON 字段缺失问题**：生成评估结果 JSON 时，直接按照 COCO 标准格式补全 `type`、`info`、`licenses` 等字段，而非修改 pycocotools 源码；可提前准备标准模板，让模型输出结果时直接填充。
- **针对图片ID缺失问题**：确保验证集所有图片都生成预测结果，不跳过任何样本；若必须跳过，需在评估时做好容错，避免程序崩溃。
- **针对编码问题**：遇到 Python2/3 兼容问题时，优先检查文件读写模式和字符串编码，对写入操作添加 `.encode('utf-8')` 转换，对读取操作添加 `.decode('utf-8')` 转换。
- **评估模块的坑很难提前发现**：建议在训练中期就尝试跑一次完整的评估流程（哪怕模型还没收敛），及早发现 JSON 格式、依赖库等问题。
- **设定止损线**：如果某个问题连续 3 天无法解决，且没有社区讨论，果断放弃，换另一个更活跃的开源项目。

### 8.2 对论文作者的建议
- **提供完整的数据准备脚本**：包括数据集下载链接、目录结构说明、路径映射脚本。
- **评估模块尽量使用标准库**：如果必须修改，请在 README 中明确说明修改点，并提供校验脚本。
- **做好 Python 版本适配**：确保代码在 Python3 环境下可正常运行，避免编码、语法等兼容问题。
- **保持 Issues 活跃**：及时响应用户的复现问题，对提高论文影响力有很大帮助。

---

## 9. 后续计划

虽然 PureT 复现失败了，但我没有放弃图像描述方向的研究。下一步计划：
- 尝试复现 **BLIP** 或 **OFA**（这两个模型的代码更活跃，社区反馈也更多），重点验证初始提出的两个问题在这些模型上是否更容易解决。
- 将重心转移到 **多模态预训练模型**，这类模型的数据准备流程通常更规范。

如果你也在复现 PureT 或遇到类似的“路径缺失/JSON字段缺失/图片ID缺失/编码兼容”问题，欢迎交流讨论！

---

**参考链接**  
- 论文原文（待补充）  
- 官方代码：[https://github.com/232525/PureT](https://github.com/232525/PureT)  
- COCO 数据集：[https://cocodataset.org/](https://cocodataset.org/)