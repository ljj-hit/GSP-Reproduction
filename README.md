# GSP-Reproduction

一个使用 **Python + PyTorch** 完成的 GSP（Generalized Search for Planning）机制级简化复现项目。项目在程序生成的 **8×8 二维迷宫**上，用 MLP 近似 Q 函数，以 **Weighted A*（WA*）** 搜索产生经验，再用 Q-learning 训练网络，并在从未参与训练和验证的 300 张测试迷宫上比较 GSP-WA* 与传统搜索算法。

> 这是一个**简化复现**，不是论文原始代码的等价实现。主要差异见[与原论文的区别](#与原论文的区别)。

---

## 1. 研究目标

项目验证的核心问题不是“如何用 BFS 解迷宫”，而是：

1. WA* 搜索过程中产生的状态—动作经验，能否训练出一个 Q 函数；
2. 该 Q 函数能否跨不同迷宫布局泛化，而不是只记住某一张地图；
3. 学到的 Q 值作为搜索优先级的一部分，能否在相同搜索预算下改变并改善后续 WA* 的搜索行为；
4. 这种搜索与学习交替（search → experience → Q-learning → better search）的机制是否能在完全独立的测试集上被定量评估。

训练迷宫、验证迷宫和测试迷宫均由程序生成，且通过布局哈希保证互不重复。测试集只用于最终评估，不参与梯度更新、模型选择或超参数调整。

---

## 2. 与原论文的区别

README、代码和结果解释必须明确以下简化：

| 方面 | 原论文 | 本项目 |
|---|---|---|
| 状态表示/网络 | Relational Graph Neural Network | 固定尺寸网格的 MLP（192→128→128→4） |
| 规划领域 | PDDL 规划领域 | 固定 8×8 四方向网格迷宫 |
| 泛化范围 | 关系结构、规模与实例泛化 | 仅验证同一 8×8 尺寸内不同地图布局的跨实例泛化 |
| 数据 | PDDL 领域数据/生成任务 | 完全程序生成的迷宫，无外部数据集 |
| 训练稳定化 | 论文原始训练配置 | 额外使用 Target Network；这是简化实现的工程选择 |
| 死路惩罚 | 搜索定义和参数依论文设置 | 固定的 `-64`，保证低于 8×8 任意简单可达路径的回报下界（最坏 ≤ -63） |
| 实例采样 | 论文的实例选择机制 | Unsolved/Solved/Satisficed 三池 1:2:4 加权采样 |

此外：

- 本项目**不声称实现跨地图尺寸泛化**。修改 `height/width` 会改变输入维度，必须重新生成数据和重训，不能直接迁移已训练模型。
- 项目中的搜索回报下界来自**搜索实际找到的路径**，不是 BFS 最短路径，也不是预先计算的最优 Q 值。
- BFS 只用于生成时的连通性/难度元数据和最终路径质量参考，**从不生成 Q-learning 标签**。
- WA* 与 GSP-WA* 的节点定义不同：传统基线按状态扩展，GSP 按状态—动作对扩展。所有算法使用同样的 128 次扩展预算，但该预算的实际含义应结合节点定义解释。

---

## 3. 环境、动作和状态编码

### 3.1 迷宫

- 形状：`8×8`，共 64 格。
- `0 = 可通行`，`1 = 障碍物`。
- 障碍物比例随机取自 `[10%, 25%]`；默认实际障碍格数为 7–16。
- 起点和终点从可通行格中随机选择，不能相同，且必须 BFS 可达。
- 每个实例保存完整障碍矩阵、`start=(row,col)`、`goal=(row,col)`、随机种子、BFS 最短路径长度、难度标签和地图哈希。

动作编号全局一致：

```text
UP = 0    (-1, 0)
DOWN = 1  ( 1, 0)
LEFT = 2  ( 0,-1)
RIGHT = 3 ( 0, 1)
```

不允许斜向移动。合法动作集合为

\[
A(s)=\{a\mid T(s,a)\text{ 在地图内且目标格可通行}\}.
\]

进入墙、越界动作均非法：`MazeEnvironment.step()` 会抛出 `InvalidActionError`，不原地停留、不产生奖励、不产生经验。合法移动奖励恒为 `-1`，到达目标后结束。

### 3.2 192 维状态

状态为 \(s=(M,p,g)\)，编码顺序固定为：

\[
X(s)=[\operatorname{flatten}(M),\operatorname{onehot}(p),\operatorname{onehot}(g)]\in\mathbb R^{192}.
\]

- `M`：8×8 障碍矩阵按行优先展平，64 维；
- `p`：当前位置 One-hot，64 维；
- `g`：目标位置 One-hot，64 维；
- dtype 为 `float32`，元素只取 0/1。

训练、搜索、验证和评估都通过 [`environment.encode_state`](environment.py) 生成该向量，避免编码顺序漂移。

---

## 4. 数据集生成

[`generator.py`](generator.py) 默认生成：

- 训练集：1000 张，master seed `20240101`；
- 验证集：200 张，master seed `20240202`；
- 测试集：300 张，master seed `20240303`。

每张地图流程：

1. 初始化全 0 的 8×8 矩阵；
2. 在该实例自己的 RNG 上抽取障碍比例，并转为障碍格数；
3. 随机设置障碍；
4. 从剩余空格中抽取起点与终点；
5. BFS 检查可达性，并拒绝最短路径小于 3 的实例；
6. 按 BFS 最短路径划分难度：`easy=3–6`、`medium=7–10`、`hard=11+`；
7. 按难度配额进行有界分层生成；配额无法满足时记录警告，再以“不限难度”的有界补齐；
8. 使用障碍矩阵 SHA-256 哈希跨三个数据集去重。

输出：

```text
data/train.npz
data/validation.npz
data/test.npz
data/dataset_manifest.json
```

NPZ 中包含 `mazes`、`starts`、`goals`、`seeds`、`bfs_lengths`、`difficulties`、`map_hashes`、`obstacle_ratios`。Manifest 记录实际数量、尝试次数、难度分布、障碍比例、BFS 长度统计和集合独立性检查。

默认数据的一次实际生成结果为：

```text
train:      1000 张，easy/medium/hard = 334/333/333
validation:  200 张，easy/medium/hard =  67/ 67/ 66
test:        300 张，easy/medium/hard = 100/100/100
跨集合重复布局：0
```

生成过程设置了最大尝试次数，不会无限循环；真正失败时会抛出明确异常，而不是静默生成不足数量。

---

## 5. Q 网络

正式主模型：

```python
QNetwork(
    input_dim=192,
    hidden_dims=(128, 128),
    output_dim=4,
)
```

对应：

```text
192 → Linear → ReLU → 128
    → Linear → ReLU → 128
    → Linear → 4
```

- 隐藏层使用 `kaiming_uniform_(nonlinearity="relu")`；
- 输出层使用 `xavier_uniform_(gain=0.1)`；
- 所有 bias 初始化为 0；
- 输出层无 Softmax/Sigmoid；
- 不使用 Dropout、BatchNorm、weight decay；
- 初始化只在模型创建时执行一次。

网络输出：

```python
[Q(s, UP), Q(s, DOWN), Q(s, LEFT), Q(s, RIGHT)]
```

形状为 `(batch_size, 4)`；单状态前向输出 `(4,)`。

---

## 6. GSP-WA*

[`search.py`](search.py) 实现了论文机制中的状态—动作节点搜索，而不调用现成 A* 库。

### 6.1 Frontier 优先级

节点是 `(s, a)`，优先级最大化：

\[
f(s,a)=g(s)+wQ_\theta(s,a),
\qquad
g(s)=-depth(s),\quad w=2.
\]

每移动一步有 \(g(s')=g(s)-1\)。使用 Python `heapq` 加负优先级实现最大堆，并用单调计数器处理同分排序，保证确定性。

### 6.2 初始化与扩展

1. 起始位置加入 Visited，深度为 0；
2. 对起点每个合法动作计算一次 Q 值，加入 Frontier；
3. 每次弹出最高优先级 `(s,a)`，执行动作得到 `s'`；
4. 若 `s'` 是目标：立即停止并回溯；
5. 若 `s'` 没有合法动作（且不是目标）：定义为真正死路；
6. 否则记录普通转移，并在首次发现 `s'` 时将它加入 Visited、生成其全部合法动作；
7. 达到 128 次状态—动作对扩展、找到目标或 Frontier 为空时停止。

遵循首次发现策略：每个状态只扩展一次，已访问状态不重新扩展，不进行父节点重连。

### 6.3 搜索经验与回报下界

普通转移：

```text
(state, action, next_state, return_bound=-inf, terminal=False, dead_end=False)
```

找到目标后沿 Parent 回溯，令最后一步 `R=-1`，倒数第二步 `R=-2`，依此类推：

```text
R_k = -(plan_length - k)
```

成功路径上的所有边都从普通记录的 `-inf` 替换为对应的有限 R，不保留冲突的重复记录。目标边自身是终止样本，训练目标为 `-1`。

真正死路：

\[
R_\bot=-64.
\]

可达的四方向无向网格中，任何可达状态都有回退边，因此真正的 `A(s')=∅` 死路在实际可达搜索树中通常不存在。本项目只保留该分支和如实计数，**不会把视觉上的死胡同人为标记为 dead-end**。

搜索期间：
- 网络切换为 `eval()`；
- 临时关闭参数 `requires_grad`；
- 全程 `torch.no_grad()`；
- 不进行任何参数更新；
- 每个新状态只前向一次，批量得到四个动作 Q 值并缓存。

---

## 7. Replay Buffer

[`replay_buffer.py`](replay_buffer.py) 是 FIFO 环形经验池：

- 容量 20000；
- 来自所有迷宫、所有 Episode 的经验进入同一个池；
- 超容量后淘汰最旧样本；
- 随机均匀抽取 64 条 Transition；
- 整理为 CUDA Tensor 的 Batch。

每条经验字段：

```python
{
    "state": ...,          # float32[192]
    "action": ...,         # 0..3
    "next_state": ...,     # float32[192]
    "return_bound": ...,   # -inf / 有限 R / -64
    "terminal": ...,       # 是否到达目标
    "dead_end": ...,       # 是否真正死路
    "next_valid_mask": ... # bool[4]，Bellman bootstrap 的合法动作 mask
}
```

---

## 8. Q-learning 训练目标

[`trainer.py`](trainer.py) 维护 Online Network \(Q_\theta\) 和 Target Network \(Q_{\theta^-}\)。初始化时 target 完全复制 online：

```python
target_model.load_state_dict(model.state_dict())
```

Target Network 不参与梯度，每 **200 次梯度更新**同步一次。

### 8.1 Bellman 估计

\[
\hat y(s,a)=-1+\gamma\max_{a'\in A(s')}Q_{\theta^-}(s',a'),\quad \gamma=1.
\]

计算最大值前对非法动作做 mask，非法动作的高 Q 值绝不会参与 bootstrap。

### 8.2 最终目标

| 样本类型 | 目标 \(y\) |
|---|---|
| 普通非终止转移 | \(\hat y\) |
| 成功路径上的非终止转移 | \(\max(R,\hat y)\) |
| 真正死路 | \(-64\) |
| 直接到达目标 | \(-1\)（不 bootstrap） |

`R` 是搜索实际找到路径的回报下界，不是 BFS 最优值。消融 “No Solution Bounds” 会禁用有限 R，使成功路径非终止样本也退化为普通 Bellman target，但终止/死路目标保持不变。

### 8.3 损失与更新

\[
L(\theta)=\frac1N\sum_{i=1}^{N}\left(Q_\theta(s_i,a_i)-y_i\right)^2.
\]

每次更新顺序：

1. 从 Replay Buffer 抽 64 条；
2. Online 网络前向，按 action 索引取对应 Q；
3. Target 网络在 `torch.no_grad()` 下计算下一状态 Q；
4. 用合法动作 mask 取 max；
5. 按样本类型构造最终 y；
6. MSE；
7. 清空梯度、反向传播；
8. `clip_grad_norm_(..., max_norm=1.0)`；
9. Adam 更新；
10. 每 200 次更新同步 target。

---

## 9. 训练闭环

**一个 Episode = 从训练集抽一张迷宫 + 执行一次完整 WA* + 收集经验 + 最多两次梯度更新。**

不是把一个 Batch 的 64 张迷宫作为 Episode。一次更新是从经验池抽样 64 条 Transition。

每个 Episode：

1. 按实例池动态采样选择一个训练迷宫；
2. 使用当前 Online Network 执行 128 预算 WA*；
3. 添加本次搜索全部经验到共享 Replay Buffer；
4. 若 Buffer 少于 64 条，不做梯度更新；
5. 否则独立抽取两个 Batch，各做一次 Q-learning 更新；
6. 根据本次结果更新实例池：
   - Unsolved：搜索失败；
   - Solved：成功，且扩展次数等于方案长度；
   - Satisficed：成功，但扩展次数大于方案长度；
7. 以 `1:2:4` 初始权重对非空类别池归一化后采样。

一个实例只会存在于最新一次结果对应的池中。

每 100 个 Episode 在固定 200 张验证迷宫上评估：
- 不反向传播；
- 不写 Replay Buffer；
- 不修改实例池；
- 使用同样 128 预算。

最佳模型选择顺序：

1. 验证成功率更高；
2. 成功率相同时，已解决实例的平均扩展次数更低；
3. 再比较平均方案长度更短。

最后一个 Episode 也会验证，并保存 best/last checkpoint。

`logs/update_log.csv` 额外记录每一次梯度更新的 Loss、Q 统计、梯度范数和 target 同步状态；`logs/training_log.csv` 则按 Episode 汇总这些量。

---

## 10. 传统算法与最终评估

[`baselines.py`](baselines.py) 和 [`evaluate.py`](evaluate.py) 实现：

1. **BFS**：无权四方向最短路径基线，也用于测试集 oracle 路径长度；
2. **A\* + Manhattan**：\(f=depth+h\)，最小优先；
3. **WA\* + Manhattan**：\(f=depth+2h\)，最小优先；
4. **Untrained-Q WA\***：固定随机种子初始化的 MLP，完全不训练，作为随机 Q 对照；
5. **GSP-WA\***：加载 `best_model.pt`，使用 \(f(s,a)=g(s)+2Q_\theta(s,a)\) 最大化；
6. **Greedy-Q（可选）**：直接取合法动作中 Q 最大的动作，带循环检测和最大步数限制。

评估要求：

- 所有算法在完全相同的 300 张测试迷宫上运行；
- 搜索扩展预算均为 128；
- 不更新参数、不写 Replay Buffer、不根据测试结果调模型；
- 记录端到端时间以及网络推理时间（GSP 和 Untrained-Q），从而区分模型推理与纯搜索控制开销；
- 分别报告所有实例扩展均值（包含失败）和成功实例扩展均值，避免隐藏失败；
- 路径质量使用 \(L_{\text{model}}/L_{\text{BFS}}\)，只对已解决实例定义，并在结果中同时报告未解决数量。

---

## 11. 指标与输出

[`metrics.py`](metrics.py) 计算：

- 成功率；
- 全部实例平均/标准差扩展数；
- 成功实例平均/标准差扩展数；
- 成功实例平均/标准差方案长度；
- 成功实例平均/标准差路径比；
- 平均/标准差端到端搜索时间；
- 平均/标准差网络推理时间；
- 未解决数量与失败原因分布。

输出：

```text
results/evaluation.csv
results/summary.json
results/figures/
```


CSV 每行是一个算法在一个测试迷宫上的运行记录，包含 `algorithm, instance_id, difficulty, bfs_length, success, expansions, plan_length, path_ratio, search_time_seconds, inference_time_seconds, non_inference_time_seconds, failure_reason, seed`。

[`visualize.py`](visualize.py) 生成独立 PNG：

1. `training_loss.png`
2. `validation_success_rate.png`
3. `validation_expansions.png`
4. `test_success_rate.png`
5. `test_expansions.png`
6. `test_path_quality.png`
7. `test_search_time.png`
8. `test_inference_time.png`（额外的推理开销图）

所有图都有标题、横纵轴名称、图例、网格和必要的误差条。使用固定且经过色觉可辨检查的分类色板；低对比度颜色对应的条形另有直接数值标签。

---

## 12. 项目结构

```text
GSP-Reproduction/
├── config.py
├── environment.py
├── generator.py
├── model.py
├── search.py
├── replay_buffer.py
├── trainer.py
├── train.py
├── baselines.py
├── evaluate.py
├── metrics.py
├── visualize.py
├── utils.py
├── check_cuda.py
├── run_experiments.py
├── requirements.txt
├── README.md
├── data/                    # 生成的 NPZ 与 manifest
├── checkpoints/             # 正式训练输出；smoke 结果在子目录
├── logs/
├── results/
│   ├── evaluation.csv
│   ├── summary.json
│   ├── ablation/
│   └── figures/
└── tests/
    ├── test_environment.py
    ├── test_generator.py
    ├── test_model.py
    ├── test_search.py
    └── test_training.py
```

所有核心超参数集中在 [`config.py`](config.py)，包括地图尺寸、障碍率、数据集数量、种子、隐藏层、学习率、Batch、回放容量、Episode 数、搜索预算、WA* 权重、target 更新间隔、GPU 设备编号和输出路径。实际运行配置会保存到：

```text
checkpoints/config.json
```

正式输出：

```text
checkpoints/best_model.pt
checkpoints/last_model.pt
checkpoints/config.json
checkpoints/training_summary.json
logs/training_log.csv
logs/validation_log.csv
logs/update_log.csv
logs/training.log
results/evaluation.csv
results/summary.json
results/figures/*.png
```

三随机种子正式实验会把每个种子隔离在 `checkpoints/seed_<seed>/`、`logs/seed_<seed>/` 和 `results/seed_<seed>/`，并在根 `results/evaluation.csv` 合并记录、在根 `results/summary.json` 报告跨种子均值和标准差。消融实验写入 `results/ablation/<name>/`，不会覆盖主实验结果。

---

## 13. GPU 要求

本项目**强制使用 NVIDIA CUDA**：

- 正式训练和模型评估要求 `torch.cuda.is_available()` 为真；
- `check_cuda.py` 会打印 GPU 名称、PyTorch CUDA 版本、设备编号、计算能力和显存；
- `QTrainer` 拒绝 CPU 设备，不会静默回退；
- 每次梯度更新前检查模型参数和各 Batch Tensor 都在指定 CUDA 设备；
- 搜索和迷宫生成可以用 CPU 控制逻辑，但 Q 网络前向与梯度更新必须在 CUDA；
- 不通过扩大模型来提高 GPU 利用率。

如果 CUDA 不可用或不匹配，程序会抛出带安装提示的异常并退出。

PyTorch 安装示例（CUDA 12.8）：

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python check_cuda.py
```

若 `requirements.txt` 先安装了 CPU wheel，请先用上面的 CUDA index 重新安装对应 `torch`。

---

## 14. 运行命令

### 14.1 标准流程

```bash
# 1. 检查 CUDA
python check_cuda.py

# 2. 生成 1000/200/300 个迷宫（首次或 --force 覆盖）
python generator.py --force

# 3. 运行测试
pytest -q

# 4. 启动 CUDA 正式训练（3000 Episodes）
python train.py --device cuda:0

# 5. 加载最佳模型进行测试
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --device cuda:0

# 6. 绘制实验图
python visualize.py
```

### 14.2 10 Episode 冒烟测试

以下命令输出到隔离目录，不污染正式的 `best_model.pt`：

```bash
python train.py \
    --device cuda:0 \
    --episodes 10 \
    --validation-interval 5 \
    --checkpoint-dir checkpoints/smoke_10ep_seed42 \
    --log-dir logs/smoke_10ep_seed42 \
    --results-dir results/smoke_10ep_seed42
```

冒烟训练通过后应停止，不应自动启动 3000 Episode 正式训练。

### 14.3 三随机种子正式实验

```bash
python run_experiments.py multi-seed \
    --device cuda:0 \
    --seeds 42 43 44
```

三个种子按顺序串行运行，避免多进程争用显存。完成后：

- 每个种子的结果在 `results/seed_<seed>/`；
- 合并的测试记录在 `results/evaluation.csv`；
- 跨种子均值/标准差在 `results/summary.json`；
- 训练曲线图使用第一个种子的训练日志，算法比较图使用所有种子合并记录（图中会注明误差条来源）。

### 14.4 统一入口

```bash
python run_experiments.py check-cuda --device cuda:0
python run_experiments.py generate --force
python run_experiments.py train --device cuda:0 --seed 42
python run_experiments.py evaluate --checkpoint checkpoints/best_model.pt --device cuda:0
python run_experiments.py visualize
python run_experiments.py all --device cuda:0 --seeds 42 43 44
```

`all` 会生成数据、串行训练、评估并绘图；运行前请确认确实希望执行完整实验。

### 14.5 消融实验

```bash
# Ablation 1：禁用成功路径的搜索回报下界
python run_experiments.py ablation \
    --name no_solution_bounds --device cuda:0 --seed 42

# Ablation 2：取消动态实例池采样，均匀抽样
python run_experiments.py ablation \
    --name uniform_sampling --device cuda:0 --seed 42

# Ablation 3：独立训练 64×64 和 128×128 两种容量
python run_experiments.py ablation \
    --name mlp_capacity --device cuda:0 --seed 42
```

输出分别保存在 `results/ablation/no_solution_bounds/`、`results/ablation/uniform_sampling/`、`results/ablation/mlp_capacity_64/` 和 `results/ablation/mlp_capacity_128/`。每组都有独立 checkpoint、日志、评估 CSV/JSON 和图表。

### 14.6 可选 Greedy-Q 评估

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --device cuda:0 \
    --include-greedy
```

主实验仍以 GSP-WA* 为准；Greedy-Q 只作为学到的 Q 函数直接决策能力的补充诊断。

---

## 15. 测试与已验证结果

### 15.1 单元测试

`pytest -q` 共 42 项测试，覆盖：

- 8×8 地图生成、起终点合法性、可达性；
- 越界/墙壁/非法动作异常且不产生状态或奖励；
- 合法移动奖励 -1、到达目标终止；
- 192 维编码、三段 One-hot 编码和通道顺序；
- MLP 结构、输出维度、初始化、Batch/单状态接口；
- checkpoint 保存后重载输出逐位一致；
- WA* 最大优先级方向、首次发现父链、路径回溯、R=-1/-2/...；
- 普通转移与成功路径有限回报的区分、重复边不冲突；
- 搜索预算失败保留普通经验、不伪造死路；
- Bellman target 的合法动作 Mask、终止不 bootstrap、死路与 `max(R, y_hat)`；
- 反向传播确实更新 Online Network；
- Target Network 严格按设定频率同步；
- Replay Buffer FIFO、采样和 CUDA Tensor；
- 实例池互斥分类；
- 实际 CUDA Batch/模型位置；
- 训练/验证/测试三个集合哈希不重叠。

实际执行结果：

```text
42 passed
```

### 15.2 本机 CUDA 检查

执行环境：

```text
CUDA available: True
NVIDIA GPU name: NVIDIA GeForce RTX 5060 Laptop GPU
PyTorch CUDA version: 12.8
PyTorch version: 2.11.0+cu128
Selected device index: 0 (cuda:0)
Compute capability: 12.0
GPU memory: 7.96 GiB
CUDA tensor smoke test: passed
```

### 15.3 10 Episode 冒烟结果

相关文件：

```text
checkpoints/smoke_10ep_seed42/best_model.pt
checkpoints/smoke_10ep_seed42/last_model.pt
checkpoints/smoke_10ep_seed42/config.json
logs/smoke_10ep_seed42/training_log.csv
logs/smoke_10ep_seed42/validation_log.csv
logs/smoke_10ep_seed42/update_log.csv
results/smoke_10ep_seed42/summary.json
results/smoke_10ep_seed42/figures/
```

观察到的现象：

- Online/Target 参数和 Batch 均位于 `cuda:0`；
- 训练损失从约 `0.93` 下降到约 `0.06`（10 个 Episode 内）；
- 5 和 10 Episode 时的验证成功率为 `78.0%` 和 `77.0%`；
- `best_model.pt` 和 `last_model.pt` 均可加载并完成 300 张测试迷宫评估；
- 完整评估和 8 张图均可生成；
- 该冒烟模型只训练了 10 个 Episode，其测试指标**不构成正式实验结果**，不能用于评价 GSP 最终性能。

正式主结果需要用户执行 3000 Episode 训练后生成到项目根输出目录。

---

## 16. 如何解读正式结果

建议按以下顺序分析：

1. **成功率**：GSP-WA* 应明显高于随机初始化 Q 网络，说明搜索经验确实改变了策略；A*/WA*/BFS 在 8×8 小地图上通常可达 100%。
2. **扩展数**：同时看 `mean_expansions_all` 和 `mean_expansions_successful`。只比较成功实例会隐藏失败，因此报告中两者都必须出现。GSP 的目标是相对 Untrained-Q WA* 改善，不保证击败强 Manhattan 启发的 WA*。
3. **路径质量**：`PathRatio` 只对已解决实例有意义。小于 1 不可能（BFS 最短），接近 1 表示路径质量好；必须结合成功率解读，因为搜索失败没有路径比。
4. **时间**：GSP 和其他学到的模型需要为每个新状态执行一次 GPU 前向，神经网络推理可能显著高于小型传统搜索的控制开销。`inference_time_seconds` 与 `non_inference_time_seconds` 分开报告。
5. **随机种子方差**：正式实验报告三种子的均值和样本标准差。单种子差异不能直接归因于算法。
6. **Ablation**：
   - 若 `no_solution_bounds` 明显降低成功率/收敛速度，说明搜索回报下界提供了有效监督；
   - 若 `uniform_sampling` 更差，说明动态实例采样有贡献；
   - 容量对比用于观察过拟合或容量不足，不保证更大模型一定更好。

8×8 迷宫较小，Manhattan WA* 非常强，BFS 展开上限也仅 64 个状态。因此本项目主要证明搜索—学习闭环与跨实例泛化机制，而不是追求在极小网格上击败经典启发式搜索。

---

## 17. 可复现性与限制

- 每个数据集 split 使用独立 master seed；每张迷宫保存实际生成种子；
- 训练、验证、测试的地图布局哈希零重叠；
- Python、NumPy、PyTorch、CUDA RNG 统一设置，并启用确定性设置；
- 所有运行参数保存到对应 checkpoint 目录的 `config.json`；
- 训练过程不读取测试集或其 BFS 最短路径；
- 禁止在测试集上做早停、调参或模型选择；
- 验证集和测试集使用同一个 128 扩展预算，但验证不写经验池；
- 本项目是机制级简化复现，不声称与论文原始算法在实现层面完全一致；
- 不声称具备跨尺寸或关系结构泛化；
- 由于四方向网格的可逆性，真正 `A(s)=∅` 死路在可达状态中通常不存在，因此死路样本数可能为 0，这是如实结果，不是程序错误。