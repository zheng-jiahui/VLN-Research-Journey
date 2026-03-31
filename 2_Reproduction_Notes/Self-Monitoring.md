# VLN-Self-Monitoring 复现全记录：踩坑、调参、结果分析

> 2025年8月 我复现了第一篇 VLN 论文 —— 《Self-Monitoring Navigation Agent via Auxiliary Progress Estimation》。当时从环境搭建到最终跑通，踩了不少坑。这篇文章就是一份完整的复现记录，希望能帮到同样在 VLN 方向入门的朋友。

---

## 1. 任务背景

**视觉-语言导航（Vision-and-Language Navigation, VLN）** 是一个非常有意思的任务：智能体需要根据自然语言指令，在一个真实的 3D 室内环境中从起点走到目标点。它既不能依赖地图，也不能预先知道目标位置的图像，只能靠视觉输入和指令之间的对齐来完成导航。

这篇论文提出的 **Self-Monitoring Agent** 主要解决了两个问题：
- 智能体不知道当前走到了指令的哪一步；
- 智能体不知道自己离目标还有多远。

为此，作者设计了两个核心模块：
1. **视觉-文本协同定位（Visual-Textual Co-Grounding）**：同时关注当前应该看哪里（视觉）和当前应该执行哪句指令（文本）；
2. **进度监控（Progress Monitor）**：估计当前已经完成了多少指令，用来约束和辅助决策。

这个模型在当时的 R2R 数据集上取得了 SOTA 结果，尤其是在 unseen 环境上提升了 8% 的成功率。

---

## 2. 环境配置

### 2.1 基础环境
- 操作系统：Ubuntu
- CUDA 版本：11.3
- Python 版本：3.7.13
- GPU：NVIDIA RTX 3090（24GB 显存）

### 2.2 依赖安装 & 踩坑记录

官方代码仓库：  
[https://github.com/chihyaoma/selfmonitoring-agent](https://github.com/chihyaoma/selfmonitoring-agent)

```bash
git clone https://github.com/chihyaoma/selfmonitoring-agent.git
cd selfmonitoring-agent
```

#### 坑 ：版本不兼容

官方 `README` 中没有明确指定 Python 和 PyTorch 的版本。我最初装了 Python 3.9 和 PyTorch 2.0，结果在运行训练脚本时频频报错，比如：

- `AttributeError: module 'torch' has no attribute 'legacy'`
- `TypeError: 'NoneType' object is not subscriptable`（来自旧版 `argparse` 用法）
- 以及一些 `numpy` 和 `torchvision` 的接口变更导致的崩溃。

查阅代码后发现，官方实现基于 **PyTorch 1.0.0** 编写，很多 API（如 `torch.legacy`、`Variable` 的隐式使用）在后续版本中被移除或修改。此外，Python 3.9 的某些语法变化也与旧代码不完全兼容。

**解决过程：**
1. 重新创建一个干净的环境，指定 Python 3.7.13（因为当时大多数复现工作都稳定在这个版本）。
2. 安装 PyTorch 1.8.0 + CUDA 11.1（实测与官方代码兼容性最好）：
   ```bash
   pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```
3. 手动修改了代码中两处与新版 PyTorch 不兼容的地方：
   - 将 `torch.legacy` 相关调用替换为显式的前向计算；
   - 修正了 `argparse` 中 `type=bool` 的用法（因为新版本不再自动转换字符串）。
4. 为了保证后续能顺利运行训练命令（例如使用 `--train_data_augmentation 1` 等参数），我统一使用了官方脚本中推荐的 `tasks/R2R-pano/main.py` 入口，并按照作者提供的示例调整了参数。

经过上述调整，代码可以正常训练，并且成功复现了论文中的基础结果。

---

## 3. 数据准备

### 3.1 数据集下载
R2R 数据集可以从以下链接获取：
- [https://github.com/peteanderson80/Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator)

下载后需要放在 `data/` 目录下，结构如下：

```
data/
├── R2R_train.json
├── R2R_val_seen.json
├── R2R_val_unseen.json
├── R2R_test.json
├── img_features/
│   └── ResNet-152-imagenet.tsv
```

### 3.2 图像特征
官方代码使用的是预提取的 ResNet-152 特征（2048 维）。  
我直接用作者提供的链接下载：

```bash
wget https://www.dropbox.com/s/xxxx/ResNet-152-imagenet.tsv  # 具体链接见官方 README
```

### 3.3 数据格式说明
每个样本包含：
- `instruction`：自然语言指令（单词列表）
- `path`：一系列视角 ID
- `trajectory`：对应的导航路径

---

## 4. 训练 / 评估过程

### 4.1 真实数据训练（baseline）

论文中最基础的模型是“co-grounding + self-monitoring”，直接在真实 R2R 训练集上训练。我使用的命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/R2R-pano/main.py \
    --exp_name 'cogrounding-selfmonitoring-agent' \
    --batch_size 64 \
    --img_fc_use_angle 1 \
    --img_feat_input_dim 2176 \
    --img_fc_dim 1024 \
    --rnn_hidden_size 512 \
    --eval_every_epochs 5 \
    --use_ignore_index 1 \
    --arch 'self-monitoring' \
    --value_loss_weight 0.5 \
    --monitor_sigmoid 0 \
    --mse_sum 0 \
    --fix_action_ended 0
```

**参数说明**：
- `--batch_size 64`：论文原始设置，但在 3090 上显存刚好够用（约 22GB）。
- `--img_feat_input_dim 2176`：图像特征（2048）+ 角度特征（128）拼接。
- `--value_loss_weight 0.5`：进度监控 loss 与 action loss 的平衡系数。
- `--eval_every_epochs 5`：每 5 个 epoch 在验证集上评估一次。

### 4.2 合成数据预训练 + 微调（提升泛化）

为了提升在 unseen 环境上的表现，论文使用了 Speaker-Follower 模型生成大量合成指令进行预训练，然后在真实数据上微调。

#### 4.2.1 合成数据预训练（300 epochs）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/R2R-pano/main.py \
    --exp_name 'cogrounding-selfmonitoring-agent' \
    --batch_size 64 \
    --img_fc_use_angle 1 \
    --img_feat_input_dim 2176 \
    --img_fc_dim 1024 \
    --rnn_hidden_size 512 \
    --eval_every_epochs 5 \
    --use_ignore_index 1 \
    --arch 'self-monitoring' \
    --value_loss_weight 0.5 \
    --monitor_sigmoid 0 \
    --mse_sum 0 \
    --fix_action_ended 0 \
    --train_data_augmentation 1 \
    --epochs_data_augmentation 300
```

- `--train_data_augmentation 1`：启用合成数据（由 Speaker 生成）。
- `--epochs_data_augmentation 300`：在合成数据上训练 300 个 epoch。

#### 4.2.2 在真实数据上微调（至多 500 epochs）

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/R2R-pano/main.py \
    --exp_name 'cogrounding-selfmonitoring-agent' \
    --batch_size 64 \
    --img_fc_use_angle 1 \
    --img_feat_input_dim 2176 \
    --img_fc_dim 1024 \
    --rnn_hidden_size 512 \
    --eval_every_epochs 5 \
    --use_ignore_index 1 \
    --arch 'self-monitoring' \
    --value_loss_weight 0.5 \
    --monitor_sigmoid 0 \
    --mse_sum 0 \
    --fix_action_ended 0 \
    --resume 'best' \
    --max_num_epochs 500 \
    --exp_name_secondary '_resume|best'
```

- `--resume 'best'`：从预训练阶段最好的模型开始微调。
- `--max_num_epochs 500`：微调最多 500 个 epoch（实际早停）。

### 4.3 训练过程中的观察

- **loss 曲线**：action selection 的交叉熵 loss 下降较快，progress monitor 的 MSE loss 下降较慢，说明进度监控更难拟合。
- **收敛时间**：大约在第 35 个 epoch 时 val seen 的 SR 达到峰值（70%），val unseen 在 40 轮左右达到 56–57%。
- **过拟合现象**：val seen 的 SR 在 45 轮后开始下降，但 val unseen 基本稳定，说明模型泛化能力还不错。
- **合成数据预训练**：预训练阶段 loss 下降非常缓慢，但微调后收敛速度明显加快，最终 unseen 成功率提升约 3%。

### 4.4 关于 Beam Search 的一点思考

论文在推理阶段使用了 **beam search**（束搜索），即同时保留多条候选路径，最后选概率最高的那条。官方提供的推理命令也包含了这一选项：

```bash
# beam search 推理
CUDA_VISIBLE_DEVICES=0 python tasks/R2R-pano/main.py \
    --exp_name 'cogrounding-selfmonitoring-agent' \
    --batch_size 64 \
    --img_fc_use_angle 1 \
    --img_feat_input_dim 2176 \
    --img_fc_dim 1024 \
    --rnn_hidden_size 512 \
    --eval_every_epochs 5 \
    --use_ignore_index 1 \
    --arch 'self-monitoring' \
    --value_loss_weight 0.5 \
    --monitor_sigmoid 0 \
    --mse_sum 0 \
    --fix_action_ended 0 \
    --resume 'best' \  # resume from best performing model
    --eval_beam 1 \  # use beam search for evaluation
    --beam_size 15  # set beam size to 15
```

但我个人认为，**beam search 在实际应用中并不合理**。它本质上是**广度优先搜索（BFS）**，会同时探索多条路径，然后在最后选一条最优的。这在评测中确实能提高成功率，但现实中的机器人一次只能走一条路，不可能同时尝试多条路线再“反悔”。除非是类似大规模搜救场景，多台机器人同时探索，否则这种“作弊式”的推断方式与真实部署脱节。

正因为如此，**绝大多数 VLN 后续工作（如 VLN-BERT、HAMT、DUET 等）都不再使用 beam search**，而是专注于设计更好的单路决策模型。我在复现时虽然也跑通了 beam search 的结果，但在最终评估时更倾向于使用 **greedy decoding** 或 **progress inference**（论文中提出的另一种单路推断方法），以保证结果的现实意义。

---

## 5. 结果

### 5.1 我在 val unseen 上复现的结果

| 方法 | SR (%) | SPL (%) |
|------|--------|--------|
| 论文原文（无数据增强） | 57.0 | 51.0 |
| 我的复现（batch size=32） | 55.2 | 49.3 |
| 我的复现（batch size=32 + 数据增强） | 58.1 | 51.8 |

注：以上结果使用 **greedy decoding** 得到，未使用 beam search。若使用 beam search，我的 SR 能达到 59% 左右，但如前所述，我认为该数值不宜作为真实能力的体现。

### 5.2 与原文对比
- 无数据增强时，我的 SR 比原文低 **1.8%**，SPL 低 **1.7%**。
- 有数据增强时，SR 达到 **58.1%**，略高于原文（58%）。

**原因分析**：
1. batch size 减半导致梯度估计不如原文稳定；
2. 原文使用了 beam search + progress monitor 的联合推断，我在复现中 beam size 调得较小（8），可能影响最终结果；
3. 随机种子差异，VLN 任务对初始化敏感。

---

## 7. 可以改进的点

如果让我在这个工作的基础上继续改进，我会考虑：

1. **引入更强的预训练模型**  
   当前使用的是 ResNet-152 + LSTM，如果换成 CLIP 或 ViT 做视觉编码，再用 BERT 做指令编码，应该能显著提升 grounding 的准确性。

2. **长指令的分段执行**  
   论文中的 attention 虽然能跟踪进度，但遇到“先做 A，再做 B，再回到 A”这种重复结构时容易乱。  
   我可能会尝试 **结构化 attention** 或 **指令分段器**，先把长指令拆成子目标。

3. **更强的进度估计信号**  
   Progress monitor 只用了归一化距离作为监督，比较粗糙。  
   可以引入 **对比学习**，让不同进度的隐状态在 embedding 空间中拉开距离，增强进度判别能力。

4. **放弃 beam search，专注单路决策**  
   正如我在 4.4 节中所述，beam search 在实际场景中不可行。后续工作可以探索更鲁棒的**单路决策策略**，例如引入**状态回溯机制**或**动态规划**，在保持实时性的同时提高成功率。

---

## 写在最后

这篇论文是我在 VLN 方向复现的第一篇工作，代码质量很高，思路也很清晰。虽然过程中踩了不少环境的坑，但跑通之后对整个任务的流程、注意力机制的设计、进度估计的作用都有了更深的理解。

如果你也在复现这篇论文，欢迎交流讨论！

