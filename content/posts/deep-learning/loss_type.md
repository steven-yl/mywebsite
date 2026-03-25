---
title: "Loss Functions：系统化整理"
subtitle: ""
date: 2026-03-25T00:00:00+08:00
draft: false
authors: [Steven]
description: "系统梳理深度学习常见任务中的损失函数：回归、分类、生成、排序、对比学习、检测、分割、序列建模与强化学习等。"
summary: "本笔记从任务视角覆盖主流 Loss Functions，包括经典方法、现代变体以及实际组合策略，便于快速对照与选型。"
tags: ["Deep Learning", "Loss Functions"]
categories: [Deep Learning]
series: [Deep Learning系列]
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

## 1. 回归任务

回归任务的目标是预测连续数值，损失函数衡量预测值 \(\hat{y}\) 与真实值 \(y\) 之间的差异。

### 1.1 均方误差 (Mean Squared Error, MSE)

- **定义**：计算预测值与真实值差值的平方的平均值。
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
- **特点**：
  - 对离群点非常敏感（误差被平方放大）。
  - 凸函数，优化简单，梯度稳定。
  - 当噪声服从高斯分布时，MSE 对应最大似然估计。
- **适用场景**：大多数普通回归问题，数据分布平滑、无显著异常值时。

### 1.2 平均绝对误差 (Mean Absolute Error, MAE)

- **定义**：计算预测值与真实值差值的绝对值的平均值。
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]
- **特点**：
  - 对离群点鲁棒（误差线性增长）。
  - 在误差接近 0 时梯度恒定，收敛较慢。
  - 预测值趋向于条件中位数。
- **适用场景**：数据中存在较多异常值，或期望模型预测中位数时。

### 1.3 Huber Loss

- **定义**：结合 MSE 与 MAE 的优点，通过阈值 \(\delta\) 切换。
  \[
  L_{\delta}(y, \hat{y}) = 
  \begin{cases}
  \frac{1}{2}(y - \hat{y})^2, & |y - \hat{y}| \le \delta \\
  \delta \left( |y - \hat{y}| - \frac{1}{2}\delta \right), & \text{otherwise}
  \end{cases}
  \]
- **特点**：
  - 小误差时平滑可导（MSE），大误差时线性（MAE）。
  - 需手动调整超参数 \(\delta\)（通常设为 1 或根据数据尺度调整）。
- **适用场景**：兼顾离群点鲁棒性与收敛稳定性，回归任务中的“万金油”选择。

---

## 2. 分类任务

分类任务的目标是将输入分配到离散的类别中。输出可以是概率（交叉熵系列）或分数（合页损失）。

### 2.1 二元交叉熵 (Binary Cross Entropy, BCE)

- **定义**：用于二分类，衡量真实分布 \(y \in \{0,1\}\) 与预测概率 \(\hat{y} \in (0,1)\) 的差异。
  \[
  \text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
  \]
- **特点**：
  - 极大似然估计的对数形式。
  - 当模型预测与真实标签相反时，损失趋向无穷大，迫使模型快速修正。
- **适用场景**：二分类任务（垃圾邮件识别、点击率预测等），输出层通常接 Sigmoid 激活函数。

### 2.2 多分类交叉熵 (Categorical Cross Entropy, CCE)

- **定义**：将二分类扩展至多分类，配合 Softmax 激活函数。
  \[
  \text{CCE} = -\sum_{i=1}^{n} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
  \]
  其中 \(y_{i,c}\) 为 one-hot 编码的真实标签，\(\hat{y}_{i,c}\) 为预测概率。
- **特点**：
  - 分类任务中最常用的损失函数。
  - 鼓励模型提高正确类别的概率，压制错误类别的概率。
- **适用场景**：互斥多分类问题（图像分类、文本分类等）。

### 2.3 合页损失 (Hinge Loss)

- **定义**：用于支持向量机（SVM）等最大间隔分类器。设分类器输出的分数为 \(s\)，则对于样本 \(i\)，真实类别 \(y_i\)，其他类别 \(j\) 的分数 \(s_j\)，损失为：
  \[
  \text{Hinge} = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)
  \]
  通常取 \(\Delta = 1\)。
- **特点**：
  - 追求“间隔最大化”，使正确类别分数至少比错误类别高出边际 \(\Delta\)。
  - 对异常点相对鲁棒，因为超出边际后不再进一步惩罚。
- **适用场景**：支持向量机、结构化预测任务。

### 2.4 Focal Loss

- **定义**：在交叉熵基础上增加调制系数 \((1-p_t)^\gamma\)，其中 \(p_t\) 为模型对真实类别的预测概率。
  \[
  \text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)
  \]
  通常取 \(\gamma = 2\)，\(\alpha_t\) 为类别权重。
- **特点**：
  - 解决类别不平衡问题：对于分类良好的样本（\(p_t\) 高），贡献大幅降低；困难样本或少数类样本权重自动提高。
  - 使模型聚焦于难分样本。
- **适用场景**：目标检测（RetinaNet）、极度不平衡的数据集（医疗诊断、欺诈检测等）。

---

## 3. 生成任务

生成模型的目标是学习数据分布并生成新样本。损失函数通常衡量生成分布与真实分布的差异。

### 3.1 KL 散度 (Kullback-Leibler Divergence)

- **定义**：衡量两个概率分布 \(P\)（真实）和 \(Q\)（预测）之间的相对熵。
  \[
  D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
  \]
- **特点**：
  - 非对称性，不满足距离定义（\(D_{KL}(P\|Q) \neq D_{KL}(Q\|P)\)）。
  - 在生成模型中用于约束隐空间分布。
- **适用场景**：变分自编码器（VAE）、知识蒸馏。

### 3.2 GAN 损失 (Generative Adversarial Network Loss)

生成对抗网络包含两个相互博弈的损失：

#### 3.2.1 原始 GAN 损失 (Minimax Loss)
- **判别器 D**：最大化区分真实样本和生成样本的能力。
  \[
  \mathcal{L}_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
  \]
- **生成器 G**：最小化判别器识别出生成样本的概率。
  \[
  \mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]
  \]
- **特点**：博弈性质，训练不稳定，易出现模式崩塌和梯度消失。

#### 3.2.2 Wasserstein GAN 损失 (WGAN)
- **定义**：使用推土机距离（Earth Mover‘s Distance）替代 JS 散度。判别器变为 Critic，输出未归一化的分数。
  \[
  \mathcal{L}_{WGAN} = \mathbb{E}_{x \sim p_{\text{data}}}[f_w(x)] - \mathbb{E}_{z \sim p_z}[f_w(G(z))]
  \]
  约束 Critic 为 1-Lipschitz 函数（通过梯度惩罚或权重裁剪）。
- **特点**：训练更稳定，损失值与样本质量呈负相关，缓解模式崩塌。
- **适用场景**：图像生成、风格迁移、超分辨率等现代 GAN 应用。

---

## 4. 排序与检索任务

这类任务关注物品之间的相对顺序，而非绝对分数。常用于推荐系统、信息检索、度量学习。

### 4.1 成对排序损失 (Bayesian Personalized Ranking, BPR)

- **定义**：基于用户交互数据，假设正样本（点击/购买）得分应高于负样本（未点击）。
  \[
  \mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in \mathcal{D}} \log \sigma(x_{u,i} - x_{u,j})
  \]
  其中 \(x_{u,i}\) 为用户 \(u\) 对物品 \(i\) 的预测得分，\(\sigma\) 为 Sigmoid 函数。
- **特点**：优化的是相对顺序而非绝对分数，适合隐式反馈（仅有正样本，负样本通过采样获得）。
- **适用场景**：协同过滤、搜索排序、推荐系统。

### 4.2 Triplet Loss

- **定义**：用于度量学习，将输入映射到嵌入空间。要求锚点 (Anchor) 与正样本 (Positive) 的距离比与负样本 (Negative) 的距离小至少一个边际 \(\alpha\)。
  \[
  \mathcal{L}_{\text{triplet}} = \max\left( d(a,p) - d(a,n) + \alpha, 0 \right)
  \]
  其中 \(d(\cdot,\cdot)\) 通常为欧氏距离或余弦距离。
- **特点**：能够学习出具有区分度的特征表示，需精心挑选三元组（困难负样本挖掘）。
- **适用场景**：人脸识别（FaceNet）、图像检索、Siamese 网络。

### 4.3 Margin Ranking Loss

- **定义**：用于排序任务，输入为两个样本及其相对顺序标签。
  \[
  \mathcal{L}_{\text{margin}} = \max(0, -y \cdot (x_1 - x_2) + \text{margin})
  \]
  其中 \(y \in \{+1, -1\}\) 表示 \(x_1\) 应大于 \(x_2\) 或反之。
- **特点**：支持成对比较，可处理不同形式的输入。
- **适用场景**：排序学习、相似性学习。

---

## 5. 对比学习与自监督学习

对比学习通过在嵌入空间中拉近正样本对、推远负样本对来学习表示，是自监督预训练的核心技术。

### 5.1 InfoNCE Loss

- **定义**：Noise Contrastive Estimation 的变体。在多个正负样本对中，最大化正样本对之间的互信息。
  \[
  \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\exp(q \cdot k_+ / \tau) + \sum_{k_- \in \mathcal{N}} \exp(q \cdot k_- / \tau)}
  \]
  其中 \(q\) 为查询（锚点），\(k_+\) 为正样本，\(\mathcal{N}\) 为负样本集合，\(\tau\) 为温度系数。
- **特点**：
  - 温度系数 \(\tau\) 控制对困难负样本的关注程度（\(\tau\) 越小，惩罚越集中）。
  - 对比学习框架（SimCLR、MoCo、CLIP）的基石。
- **适用场景**：视觉自监督预训练、多模态对齐、图神经网络预训练。

### 5.2 余弦相似度损失 (Cosine Similarity Loss)

- **定义**：衡量两个向量在方向上的相似度，忽略模长影响。
  \[
  \mathcal{L}_{\text{cosine}} = 1 - \cos(\theta) = 1 - \frac{A \cdot B}{\|A\| \|B\|}
  \]
  也可结合边际使用：\(\max(0, \cos(A,B_{\text{neg}}) - \cos(A,B_{\text{pos}}) + \alpha)\)。
- **特点**：对向量模长不敏感，适合归一化后的特征匹配。
- **适用场景**：人脸识别（结合边际）、文本相似度计算、检索系统。

---

## 6. 目标检测专用损失

目标检测任务通常包含分类与定位两个子任务，因此损失函数为二者加权和。定位部分已从传统的 L1/L2 演变为 IoU 系列损失。

### 6.1 边界框回归损失 (IoU 系列)

#### 6.1.1 IoU Loss
- **定义**：直接优化预测框与真实框的交并比。
  \[
  \mathcal{L}_{\text{IoU}} = 1 - \frac{|A \cap B|}{|A \cup B|}
  \]
- **特点**：当两个框不相交时，梯度为 0，无法优化。

#### 6.1.2 GIoU Loss (Generalized IoU)
- **定义**：引入最小外接框 \(C\)，解决不相交时梯度消失问题。
  \[
  \mathcal{L}_{\text{GIoU}} = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|}
  \]
- **特点**：即使不相交，也能提供有意义的梯度。

#### 6.1.3 DIoU Loss (Distance IoU) 与 CIoU Loss (Complete IoU)
- **DIoU**：在 GIoU 基础上加入中心点距离惩罚。
  \[
  \mathcal{L}_{\text{DIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2}
  \]
  其中 \(\rho\) 为欧氏距离，\(c\) 为最小外接框对角线长度。
- **CIoU**：进一步加入长宽比一致性惩罚。
  \[
  \mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
  \]
  其中 \(v\) 衡量长宽比差异，\(\alpha\) 为平衡系数。
- **特点**：现代检测器（YOLO 系列、Faster R-CNN）的标准配置，收敛更快、定位更准。

### 6.2 分类损失

目标检测的分类分支通常使用交叉熵或 Focal Loss，以应对前景-背景严重不平衡的问题。Focal Loss 在单阶段检测器中尤为常用。

### 6.3 综合损失

典型目标检测器的总损失为：
\[
\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}}
\]
其中 \(\mathcal{L}_{\text{cls}}\) 为分类损失（交叉熵或 Focal），\(\mathcal{L}_{\text{box}}\) 为边界框回归损失（CIoU 等），\(\mathcal{L}_{\text{obj}}\) 为目标置信度损失（通常为二元交叉熵）。

---

## 7. 图像分割专用损失

图像分割任务需要对每个像素进行分类，面临类别不平衡、小目标难分割等问题。

### 7.1 Dice Loss

- **定义**：直接优化预测分割与真实分割的 Dice 系数。
  \[
  \mathcal{L}_{\text{Dice}} = 1 - \frac{2|X \cap Y|}{|X| + |Y|}
  \]
  其中 \(X\) 为预测区域，\(Y\) 为真实区域。
- **特点**：
  - 天然应对类别不平衡，尤其适用于前景占比极小的场景。
  - 对预测概率的置信度敏感，可与交叉熵组合使用。
- **适用场景**：医学图像分割（病灶、器官分割）、遥感图像分割。

### 7.2 交叉熵与 Dice 的组合

实践中常将交叉熵与 Dice Loss 加权求和：
\[
\mathcal{L}_{\text{comb}} = \alpha \mathcal{L}_{\text{CE}} + (1-\alpha) \mathcal{L}_{\text{Dice}}
\]
既保证像素级分类精度，又兼顾前景区域的重叠度。

### 7.3 Tversky Loss

- **定义**：Dice Loss 的推广，通过调整假阳性和假阴性的权重来处理不平衡。
  \[
  \mathcal{L}_{\text{Tversky}} = 1 - \frac{|X \cap Y|}{|X \cap Y| + \beta |X \setminus Y| + (1-\beta) |Y \setminus X|}
  \]
- **特点**：当 \(\beta = 0.5\) 时退化为 Dice，当 \(\beta > 0.5\) 时更关注假阴性（漏检），适合不同偏好。
- **适用场景**：医学分割，可根据临床需求调节漏检与误检的惩罚。

---

## 8. 结构化预测与序列生成

序列生成（文本、语音等）任务中，输入与输出长度可能不对齐，且评价指标不可微。

### 8.1 联结主义时间分类损失 (Connectionist Temporal Classification, CTC)

- **定义**：用于解决输入序列与输出序列长度不对齐且未对齐的问题。CTC 引入“空白符”和动态规划，自动寻找最优对齐路径。
  \[
  \mathcal{L}_{\text{CTC}} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p(\pi_t | x)
  \]
  其中 \(\mathcal{B}\) 为映射函数（移除重复和空白符），\(\pi\) 为对齐路径。
- **特点**：不需要预先标注输入帧对应的输出字符，训练端到端。
- **适用场景**：语音识别（语音波形转文字）、手写文字识别。

### 8.2 交叉熵与 Teacher Forcing

在文本生成任务（如机器翻译）中，训练阶段通常采用 Teacher Forcing，即用真实历史 token 作为输入，预测下一个 token，损失为逐 token 的交叉熵：
\[
\mathcal{L}_{\text{CE}} = -\sum_{t=1}^{T} \log p(y_t | y_{<t}, x)
\]
- **特点**：训练稳定，但存在“暴露偏差”（训练时使用真实 token，推理时使用模型自身预测）。
- **适用场景**：Seq2Seq 模型、Transformer 训练。

### 8.3 直接优化评估指标（强化学习）

由于 BLEU、ROUGE 等指标不可微，无法直接通过交叉熵优化。研究者使用强化学习（如 Self-Critical Sequence Training）直接最大化评估指标：
\[
\mathcal{L}_{\text{RL}} = -\mathbb{E}_{y \sim p_{\theta}} [r(y)] \cdot \log p_{\theta}(y)
\]
- **特点**：缓解暴露偏差，使模型在推理时表现更优。
- **适用场景**：文本摘要、对话生成、机器翻译的后续优化。

### 8.4 对比学习用于文本生成 (SimCTG)

- **定义**：在生成过程中引入对比损失，鼓励模型生成多样化的文本，避免重复。
  \[
  \mathcal{L}_{\text{SimCTG}} = -\log \frac{\exp(\text{sim}(h_{y_t}, h_{y_t^+})/\tau)}{\sum_{y^- \in \mathcal{N}} \exp(\text{sim}(h_{y_t}, h_{y^-})/\tau)}
  \]
- **特点**：有效提升生成文本的多样性和质量。
- **适用场景**：大语言模型的微调，解决重复生成问题。

---

## 9. 强化学习损失

强化学习的目标是学习策略以最大化累积奖励。损失函数通常基于策略梯度或价值函数。

### 9.1 策略梯度损失 (Policy Gradient)

- **定义**：REINFORCE 算法中的梯度估计。
  \[
  \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) R(\tau) \right]
  \]
  损失形式为：
  \[
  \mathcal{L}_{\text{PG}} = -\mathbb{E} \left[ \log \pi_{\theta}(a_t | s_t) \cdot \hat{A}_t \right]
  \]
  其中 \(\hat{A}_t\) 为优势函数估计。
- **特点**：直接优化策略，但方差较大。

### 9.2 PPO Clip Loss (Proximal Policy Optimization)

- **定义**：限制策略更新幅度，提高稳定性。
  \[
  \mathcal{L}_{\text{PPO}} = \mathbb{E} \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
  \]
  其中 \(r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\)。
- **特点**：是目前最常用的策略优化算法，平衡探索与稳定。
- **适用场景**：大语言模型的 RLHF（基于人类反馈的强化学习）、机器人控制。

### 9.3 DQN TD Loss (Temporal Difference)

- **定义**：用于值函数学习，最小化贝尔曼误差。
  \[
  \mathcal{L}_{\text{DQN}} = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
  \]
- **特点**：使用目标网络和回放池稳定训练。
- **适用场景**：离散动作空间的深度 Q 学习。

---

## 10. 图神经网络损失

图神经网络处理非欧结构数据，损失函数需适应节点、边、图级别任务。

### 10.1 节点分类损失

通常使用交叉熵（半监督学习）：
\[
\mathcal{L}_{\text{node}} = -\sum_{i \in \mathcal{V}_L} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}
\]
其中 \(\mathcal{V}_L\) 为有标签节点集合。

### 10.2 图对比学习损失 (Graph Contrastive Loss)

- **定义**：通过数据增强生成两个视图，最大化同一节点在不同视图中的表示一致性。
  \[
  \mathcal{L}_{\text{GCL}} = -\log \frac{\exp(\text{sim}(z_i, z_i')/\tau)}{\sum_{j \neq i} \exp(\text{sim}(z_i, z_j)/\tau)}
  \]
- **特点**：自监督学习图表示，无需标签。
- **适用场景**：图预训练（GraphCL、SimGRACE）。

### 10.3 边预测损失 (Edge Prediction Loss)

用于链接预测任务，通常为二元交叉熵：
\[
\mathcal{L}_{\text{edge}} = -\sum_{(u,v) \in \mathcal{E}^+} \log \sigma(\text{score}_{u,v}) - \sum_{(u,v) \in \mathcal{E}^-} \log (1 - \sigma(\text{score}_{u,v}))
\]
其中正边来自真实图，负边通过采样获得。

---

## 11. 3D 视觉与点云损失

3D 任务（点云重建、配准）需要处理无序点集，常用距离度量。

### 11.1 Chamfer Distance

- **定义**：衡量两个点集 \(P\) 和 \(Q\) 之间的差异。
  \[
  \text{CD}(P, Q) = \frac{1}{|P|} \sum_{p \in P} \min_{q \in Q} \|p - q\|_2^2 + \frac{1}{|Q|} \sum_{q \in Q} \min_{p \in P} \|q - p\|_2^2
  \]
- **特点**：计算效率高，可微，但可能忽略点云的结构信息。
- **适用场景**：点云重建、生成、形状补全。

### 11.2 Earth Mover‘s Distance (EMD)

- **定义**：求解将点云 \(P\) 变换为点云 \(Q\) 的最小搬运代价。
  \[
  \text{EMD}(P, Q) = \min_{\phi: P \to Q} \sum_{p \in P} \|p - \phi(p)\|_2
  \]
- **特点**：对点云分布更敏感，但计算复杂度高。
- **适用场景**：点云匹配、生成模型。

---

## 12. 时间序列预测损失

时间序列预测除了常规回归损失外，还有一些特定损失。

### 12.1 Quantile Loss (分位数损失)

- **定义**：用于概率预测，优化特定分位数 \(\tau\)。
  \[
  \mathcal{L}_{\tau}(y, \hat{y}) = \sum_{i} \max(\tau (y_i - \hat{y}_i), (\tau - 1)(y_i - \hat{y}_i))
  \]
- **特点**：可预测分位数区间，得到预测区间。
- **适用场景**：金融风险预测、不确定性建模。

### 12.2 sMAPE (对称平均绝对百分比误差)

- **定义**：适用于尺度差异大的序列。
  \[
  \text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
  \]
- **特点**：对异常值稳健，且与尺度无关。
- **适用场景**：零售销量预测、能源负荷预测。

### 12.3 Pinball Loss

分位数损失的另一种形式，等价于 Quantile Loss。

---

## 13. 特殊优化技巧

这些技巧并非独立的损失函数，而是对现有损失函数的改进，广泛用于提升模型泛化能力、训练稳定性。

### 13.1 标签平滑 (Label Smoothing)

- **做法**：将硬标签（如 one-hot）替换为软标签：
  \[
  y'_i = (1 - \epsilon) y_i + \frac{\epsilon}{K}
  \]
  其中 \(K\) 为类别数，\(\epsilon\) 为平滑系数（通常取 0.1）。
- **效果**：防止模型对预测结果过于自信，减少过拟合，提升泛化能力和对抗鲁棒性。现代大模型训练的标准配置。

### 13.2 中心损失 (Center Loss)

- **定义**：在分类损失基础上加入惩罚项，使同一类别的特征向量在嵌入空间中靠近类中心。
  \[
  \mathcal{L} = \mathcal{L}_{\text{cls}} + \frac{\lambda}{2} \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2
  \]
  其中 \(c_{y_i}\) 为类别中心，随训练更新。
- **效果**：缩小类内距离，扩大类间距离，提升判别能力。
- **适用场景**：人脸识别、细粒度图像分类。

---

## 14. 损失函数组合与任务选择指南

实际工程中，损失函数往往是多种损失的加权组合。以下给出常见任务的标准配置及组合示例。

### 14.1 常见任务推荐损失

| 任务类型 | 推荐损失 | 理由 |
|----------|----------|------|
| 连续值预测（无异常值） | MSE | 凸性，易优化 |
| 连续值预测（有异常值） | MAE 或 Huber | 鲁棒性 |
| 二分类 | 二元交叉熵 | 概率解释，标准选择 |
| 多分类（互斥） | 多分类交叉熵 + 标签平滑 | 配合 Softmax |
| 多标签分类 | 二元交叉熵（独立 sigmoid） | 各标签独立 |
| 类别极度不平衡 | Focal Loss | 聚焦困难样本 |
| 排序/检索 | Triplet，BPR，Margin Ranking | 优化相对顺序 |
| 自监督预训练 | InfoNCE | 对比学习框架 |
| 目标检测 | CIoU + 交叉熵 + Focal（可选） | 综合回归与分类 |
| 图像分割 | Dice + 交叉熵 | 平衡前景与背景 |
| 语音识别 | CTC | 处理未对齐序列 |
| 文本生成（训练） | 交叉熵 + 标签平滑 | 稳定训练 |
| 文本生成（后优化） | 强化学习（BLEU/ROUGE） | 缓解暴露偏差 |
| 图节点分类 | 交叉熵 | 半监督学习 |
| 图自监督 | 图对比损失 | 无需标签 |
| 点云重建 | Chamfer Distance 或 EMD | 衡量点集差异 |
| 强化学习 | PPO Clip Loss | 稳定策略优化 |

### 14.2 现代模型中的组合损失示例

- **目标检测（YOLOv8）**：
  \[
  \mathcal{L} = \lambda_{\text{cls}} \cdot \text{BCE} + \lambda_{\text{box}} \cdot \text{CIoU} + \lambda_{\text{obj}} \cdot \text{BCE}
  \]

- **对比学习（SimCLR）**：
  \[
  \mathcal{L} = \text{InfoNCE (NT-Xent)}
  \]

- **多模态对齐（CLIP）**：
  \[
  \mathcal{L} = \frac{1}{2} \left( \text{InfoNCE}_{\text{image}\to\text{text}} + \text{InfoNCE}_{\text{text}\to\text{image}} \right)
  \]

- **大语言模型 SFT**：
  \[
  \mathcal{L} = \text{交叉熵} + \text{标签平滑}
  \]

- **大语言模型 RLHF**：
  \[
  \mathcal{L} = \text{PPO Clip Loss} \quad \text{或} \quad \text{DPO Loss}
  \]

- **医学图像分割**：
  \[
  \mathcal{L} = \alpha \cdot \text{交叉熵} + (1-\alpha) \cdot \text{Dice Loss}
  \]

- **人脸识别**：
  \[
  \mathcal{L} = \text{交叉熵（带边际）} + \lambda \cdot \text{中心损失}
  \]

---

## 总结

本笔记系统梳理了机器学习与深度学习中几乎全部主流损失函数，涵盖回归、分类、生成、排序、对比学习、目标检测、分割、序列建模、强化学习、图网络、3D视觉、时间序列等各个领域，并补充了标签平滑、中心损失等优化技巧以及实际组合示例。

**选择损失函数的核心原则**：
1. 理解任务本质（回归/分类/排序/生成等）及输出形式。
2. 考虑数据特性（噪声分布、不平衡性、尺度等）。
3. 优先使用领域内验证过的标准损失，遇到瓶颈时再尝试变体或组合。
4. 损失函数只是目标函数的一部分，与模型结构、优化器、超参数共同决定最终效果。
