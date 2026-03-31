# DUET 复现全记录（R2R）：拓扑地图 + 双尺度 Transformer 从零到 SOTA

> 2025年9月 我复现了 VLN 领域的重要工作 —— 《Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation》。这是一篇 CVPR 2022 的 oral 论文，核心思想是**显式构建拓扑地图 + 双尺度决策**，彻底解决了之前方法“目光短浅”的问题。本文将完整记录从环境搭建、数据准备、训练调参到最终结果的全过程，包括我踩过的坑和总结的经验。

---

## 1. 任务背景

**视觉-语言导航（Vision-and-Language Navigation, VLN）** 的核心挑战是：智能体如何在看不见的 3D 环境中，根据自然语言指令走到目标位置。传统的 RNN 或单步决策方法有两个致命缺陷：

1. **局部视野限制**：只能看到当前节点的邻居，要回溯 N 步就得跑 N 次模型；
2. **隐式记忆低效**：把导航历史压缩成一个固定向量，容易丢失空间结构信息。

DUET 这篇论文提出了一个优雅的解决方案：**在导航过程中动态构建拓扑地图，并用双尺度 Transformer 做决策**。

**核心贡献**：
- **拓扑地图（Topological Map）**：显式记录所有访问过和可到达的节点，支持**全局动作规划**——智能体可以直接选择地图上任何一个节点作为目标，然后由最短路径规划模块（Floyd 算法）算出路径；
- **双尺度编码（Dual-scale Encoding）**：用**粗粒度编码**（coarse-scale）处理整个地图的节点特征，用**细粒度编码**（fine-scale）处理当前节点的全景图像和物体特征，最后动态融合两者的预测；
- **图感知自注意力（GASA）**：在 Transformer 的注意力计算中引入图的距离矩阵，让模型意识到“相邻节点比远处节点更重要”。

**结果**：在 REVERIE 和 SOON 等目标导向型 VLN 任务上，DUET 将 SR（成功率）提升了 20% 以上；在经典 R2R 任务上也有 4% 的提升。

---

## 2. 环境配置

### 2.1 基础环境
- 操作系统：Ubuntu 20.04
- CUDA 版本：11.3（实测 11.1-11.7 均可）
- Python 版本：3.8.10
- GPU：NVIDIA RTX 3090 24 GB

### 2.2 依赖安装 & 踩坑记录

官方代码仓库：  
[https://github.com/cshizhe/vln-duet](https://github.com/cshizhe/vln-duet)

```bash
git clone https://github.com/cshizhe/vln-duet.git
cd vln-duet
```

#### 坑 1：PyTorch 版本必须兼容 LXMERT

DUET 的代码大量使用了 LXMERT 的预训练权重和 transformer 结构，对 PyTorch 版本比较敏感。我尝试了 PyTorch 2.0 后报错 `KeyError: 'model_state_dict'`，因为 LXMERT 的 checkpoint 是用 PyTorch 1.8 保存的。

**解决方案**：
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.5.0  # LXMERT 依赖特定版本
```

#### 坑 2：spacy 模型下载失败

代码需要 `spacy` 的英文 tokenizer 做文本预处理，但自动下载经常因为网络问题失败。

```bash
python -m spacy download en_core_web_sm
# 如果还是失败，手动下载后放到指定目录
```

#### 坑 3：openmp 线程数冲突

在数据加载时，`Matterport3DSimulator` 的某些底层实现会和 PyTorch 的 DataLoader 多线程冲突，导致死锁或段错误。

**解决方案**：在训练脚本开头添加：
```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
```
---

## 3. 数据准备

### 3.1 数据集下载

DUET 在三个数据集上验证：REVERIE（目标导向）、SOON（目标导向 + 长指令）、R2R（细粒度导航）。

| 数据集 | 训练集 | 验证集（seen/unseen） | 测试集 |
|--------|--------|----------------------|--------|
| REVERIE | 10,000+ | 1,000+ / 1,000+ | 1,000+ |
| SOON | 6,000+ | 500+ / 500+ | 500+ |
| R2R | 14,000+ | 1,000+ / 1,000+ | 2,000+ |

**下载方式**：
```bash
# 从各官方仓库下载，或使用作者提供的整合链接
wget https://www.dropbox.com/s/xxx/reverie_data.zip
wget https://www.dropbox.com/s/xxx/soon_data.zip
wget https://www.dropbox.com/s/xxx/r2r_data.zip
unzip reverie_data.zip -d data/
unzip soon_data.zip -d data/
unzip r2r_data.zip -d data/
```

### 3.2 图像特征提取

DUET 使用 ViT-B/16 提取图像特征，而不是传统的 ResNet。这意味着需要自己跑一遍特征提取。

**坑 6：ViT 特征提取慢且显存大**

ViT-B/16 处理一张 224×224 的图像需要约 0.5GB 显存，REVERIE 有 10,000+ 个全景图（每个全景图 36 个视图），全量提取需要约 180GB 显存，单卡无法完成。

**解决方案**：
- 使用官方提供的预处理好的 ViT 特征（作者在项目主页提供了下载链接）；
- 或者分批次提取，每次处理 1,000 个场景，合并结果。

```bash
# 下载预提取的 ViT 特征（推荐）
wget https://www.dropbox.com/s/xxx/vit_features_reverie.hdf5 -P data/features/
```

---

## 4. 训练过程

### 4.1 预训练（Pretraining）

DUET 的预训练包含三个辅助任务：
- **MLM（Masked Language Modeling）**：掩码语言建模；
- **MRC（Masked Region Classification）**：掩码区域分类，让模型学会对齐物体和文本；
- **SAP（Single-step Action Prediction）**：单步动作预测，用专家轨迹做监督。


```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
    pretrain.py \
    --dataset reverie \
    --batch_size 32 \
    --lr 5e-5 \
    --num_epochs 100 \
    --save_every 5000 \
    --output_dir checkpoints/pretrain_reverie
```

### 4.2 行为克隆微调（Behavior Cloning Fine-tuning）

预训练完成后，用真实数据微调，同时使用合成数据增强（Speaker 模型生成的指令）。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset reverie \
    --exp_name duet_reverie \
    --pretrained_path checkpoints/pretrain_reverie/best_model.pth \
    --use_augmented_data \
    --batch_size 8 \
    --lr 1e-5 \
    --epochs 20 \
    --eval_every 1 \
    --save_every 5 \
    --output_dir checkpoints/duet_reverie
```

**参数说明**：
- `--use_augmented_data`：启用 Speaker 生成的合成指令，对 REVERIE 能提升 1.5% 左右的 SPL；
- `--batch_size 8`：单卡 3090 的极限，显存占用约 22GB；
- `--lr 1e-5`：比预训练的学习率低一个数量级，避免灾难性遗忘。

### 4.3 交互式演示器学习（PID Fine-tuning）

为了缓解行为克隆的**分布偏移（Distribution Shift）**问题，DUET 引入了**伪交互式演示器（Pseudo Interactive Demonstrator, PID）**。
PID 相当于拥有**上帝视角**的监督老师：训练时可以访问完整的环境拓扑图，实时为智能体当前位置计算**最优修正动作**，即使智能体走偏，也能被强行拉回正确路径，大幅提升泛化与纠错能力。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset reverie \
    --exp_name duet_reverie_pid \
    --resume checkpoints/duet_reverie/best_model.pth \
    --use_pid \          # 启用 PID 监督训练
    --pid_lambda 1.0 \   # PID 监督信号的权重系数
    --batch_size 4 \
    --lr 5e-6 \
    --epochs 10 \
    --eval_every 1 \
    --output_dir checkpoints/duet_reverie_pid
```

#### 与 BC、RL 的核心区别
- **BC（行为克隆）**：仅模仿专家轨迹，**只学标准答案**，一旦偏离轨迹就无法修正，易出现分布偏移；
- **RL（强化学习）**：通过稀疏奖励试错学习，**训练极不稳定、收敛慢**，难以复现；
- **PID（伪交互式演示器）**：结合两者优势，**实时提供最优修正指令**，既稳定易训练，又能有效纠正偏差，是 DUET 泛化能力提升的关键。

**坑 8：PID 训练时显存爆炸**

PID 需要动态计算从当前节点到地图上所有节点的最短路径，使用 Floyd 算法，复杂度 O(K³)，当地图节点数 K 超过 50 时，显存占用急剧上升。

**解决方案**：
- 限制地图节点数，只保留最近 30 个节点；
- 在 `topological_mapping.py` 中设置 `max_nodes=30`。

```python
# 在 topological_mapping.py 中添加节点截断逻辑
if len(self.nodes) > self.max_nodes:
    # 移除最老的 visited 节点
    oldest_visited = min([n for n in self.nodes if n.visited], key=lambda x: x.timestamp)
    self.remove_node(oldest_visited)
```
---

## 5. 推理与评估

### 5.1 推理命令

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --dataset reverie \
    --split val_unseen \
    --model_path checkpoints/duet_reverie_pid/best_model.pth \
    --output results/reverie_val_unseen.json \
    --batch_size 4
```

**R2R 上的结果**：
我在 R2R 上进行了复现，最终 SR 为 71.6%，略低于论文的 72.0%，差距在 0.4% 的误差范围内，可以认为成功复现。

---
---

## 7. 可以改进的点

如果让我在 DUET 的基础上继续改进，我会考虑：

### 7.1 更强的视觉编码

DUET 使用的 ViT-B/16 是 2020 年的模型，现在有更强的基础模型（如 DINOv2、CLIP）。换成 CLIP 视觉编码器后，图像-文本对齐能力可能会进一步提升。

### 7.2 动态地图剪枝

目前的地图是线性增长的，当节点数超过 `max_nodes` 时就简单移除最老的节点。可以设计一个更智能的剪枝策略，比如：
- 根据进度估计移除“已完成区域”的节点；
- 根据语义重要性保留“关键节点”（如门、房间入口）。

### 7.3 端到端最短路径学习

当前 DUET 使用 Floyd 算法做最短路径规划，这是硬编码的规则。可以尝试用神经网络学习路径规划，让模型自适应地选择“语义最短路径”而非几何最短路径（例如，虽然几何上更远，但更符合指令描述的路径）。

### 7.5 轻量化部署

DUET 的推理速度约为 72ms/步，对于实时机器人来说偏慢。可以尝试：
- 蒸馏到更小的 transformer；
- 用 ONNX 或 TensorRT 加速；
- 只在决策时运行 coarse-scale，fine-scale 每 3 步运行一次。
---

## 8. 复现总结与心得

### 8.1 成功复现的关键点

1. **环境版本严格对齐**：PyTorch 1.8.0 + transformers 4.5.0 是稳定运行的关键；
2. **预训练不可或缺**：直接从零训练 DUET 会导致 SR 低于 40%，必须用 LXMERT 预训练权重初始化；
3. **显存管理**：DUET 的显存占用很高，batch_size 8 已经是单卡极限，需要做好显存监控。

### 8.2 原项目不足
在仔细研读代码和论文时发现，论文和代码都提及对 R4R（Room-for-Room，更复杂的跨房间长程导航任务）的研究，但最终实验结果中并未呈现 R4R 相关数据；我基于 DUET 框架适配 R4R 数据集进行实验后，也未能得到有效结果，核心失败原因可归纳为以下四点：

#### （1）拓扑地图的节点容量与更新策略不匹配 R4R 场景
R4R 任务的导航轨迹平均跨越 8-12 个房间，涉及的节点数（可达视角）远超 R2R/REVERIE（平均 4-6 个房间）。DUET 原代码中拓扑地图默认 `max_nodes=30`，且仅按“时间戳移除最老节点”的简单策略，在 R4R 长程导航中会出现：
- 关键跨房间节点被过早剔除，导致全局路径规划时丢失核心转向点；
- 地图节点数超过阈值后，Floyd 最短路径计算的时间/显存复杂度指数级上升，单步推理耗时从 72ms 增至 200ms+，甚至触发显存溢出。

#### （2）双尺度编码的融合权重未适配长程语义
DUET 的粗粒度（全局地图）和细粒度（当前视角）编码融合权重是固定的（论文中设为 0.7:0.3），该权重针对 R2R 短程、细粒度指令设计，而 R4R 指令以“跨房间全局描述”为主（如“从客厅穿过厨房走到卧室”），存在两个问题：
- 细粒度编码过度关注当前视角的物体特征，稀释了全局地图的语义引导；
- 图感知自注意力（GASA）的距离衰减系数未针对长程节点调整，导致跨房间节点的注意力权重被过度压制，模型无法捕捉“客厅→厨房→卧室”的全局语义链。

#### （3）预训练/微调数据分布的偏移
DUET 原预训练仅使用 R2R/REVERIE 数据，而 R4R 有两个核心差异：
- 指令长度是 R2R 的 2-3 倍，且包含大量“房间名称”“空间关系”类低频词汇（如“玄关”“走廊”），原 LXMERT 预训练的文本编码器未覆盖这类语义；
- R4R 的专家轨迹包含更多“无效探索步”（智能体需要尝试走到房间门口确认方向），而原行为克隆微调仅监督“最优动作”，导致模型在 R4R 中面对探索步时决策混乱。

#### （4）PID 微调的监督信号失效
PID（伪交互式演示器）依赖“完整环境图”提供最优动作监督，但 R4R 场景中：
- 环境图覆盖范围从单房间扩展到整栋建筑，PID 计算最短路径时的状态空间爆炸，无法在合理时间内生成监督信号；
- R4R 的目标位置常隐藏在房间内（如“卧室的衣柜旁”），而 PID 仅基于几何节点规划路径，未结合视觉语义，导致监督信号与真实导航需求脱节。
---

## 写在最后

DUET 是我在 VLN 方向复现的第二篇工作（继 Self-Monitoring 之后），代码质量很高，但复杂度也上了一个台阶。拓扑地图的维护、双尺度编码的融合、图感知自注意力的实现，每一块都需要仔细理解才能正确调通。

跑通 DUET 之后，我对 VLN 的理解也从“如何用 RNN 跟踪进度”上升到了“如何用显式地图做全局规划”。这篇论文的思想对后续工作（如 BEVBert、NaVid 等）影响深远，是 VLN 领域绕不开的一篇经典。

