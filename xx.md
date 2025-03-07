### **ExCB算法设计适配GCN+Attention模型的步骤**

---

#### **1. 输入与初始化**
- **输入**：
  - 图数据 \( \mathcal{G} \)（节点特征 + 邻接矩阵）
  - 批次大小 \( N_B \)，聚类数 \( K \)，动量参数 \( m_s = 0.999 \)
- **初始化**：
  - **GCN特征提取器** \( f_s \): 提取节点/图特征 \( \boldsymbol{v} \in \mathbb{R}^{d} \)
  - **投影头 \( h_s \)**：2层MLP，将 \( \boldsymbol{v} \) 映射到潜在空间 \( \boldsymbol{z}_h \in \mathbb{R}^{256} \)
  - **预测头 \( g_s \)**：1层MLP，生成 \( \boldsymbol{z}_g \in \mathbb{R}^{256} \)
  - **聚类中心 \( C_s \in \mathbb{R}^{K \times 256} \)**：随机初始化或通过K-means预计算
  - **动量聚类大小向量 \( \boldsymbol{s} \in \mathbb{R}^{K} \)**：初始化为均匀分布 \( s^{(k)} = \frac{1}{K} \)

---

#### **2. 在线聚类平衡核心算法**
##### **步骤1：生成多视图特征**
- **全局视图 \( \boldsymbol{v}_G \)**：完整图结构，经GCN提取特征 \( \boldsymbol{v}_G = f_s(\mathcal{G}) \)
- **局部视图 \( \boldsymbol{v}_L \)**：子图采样或节点遮蔽，生成 \( \boldsymbol{v}_L = f_s(\mathcal{G}_{\text{sub}}) \)

##### **步骤2：计算相似度与概率分布**
- **投影特征相似度**：
  \[
  \boldsymbol{z}_s^h = \text{softmax}\left( \frac{\boldsymbol{v}_G \cdot C_s}{\|\boldsymbol{v}_G\| \|C_s\|} \right) \quad (\text{余弦相似度 + Softmax})
  \]
- **预测特征相似度**（固定聚类中心梯度）：
  \[
  \boldsymbol{z}_s^g = \text{softmax}\left( \frac{g_s(\boldsymbol{v}_G) \cdot \overline{C_s}}{\|g_s(\boldsymbol{v}_G)\| \|\overline{C_s}\|} \right) \quad (\overline{C_s} \text{为停止梯度的聚类中心})
  \]

##### **步骤3：动量更新聚类大小**
- **硬分配统计**：
  \[
  s_B^{(k)} = \frac{1}{N_B} \sum_{n=1}^{N_B} \mathbb{1}\left( \arg\max(\boldsymbol{z}_t^{(n)}) = k \right)
  \]
  （\( \boldsymbol{z}_t \) 为教师模型输出的相似度）
- **动量更新**：
  \[
  \boldsymbol{s} \leftarrow m_s \cdot \boldsymbol{s} + (1 - m_s) \cdot \boldsymbol{s}_B
  \]

##### **步骤4：平衡操作符 \( \mathcal{B} \) 调整相似度**
- **对每个聚类 \( k \)**：
  \[
  z_t^{B(k)} = 
  \begin{cases} 
  1 - [1 - z_t^{(k)}] \cdot (s^{(k)} K) & \text{if } s^{(k)} < \frac{1}{K} \\
  [1 + z_t^{(k)}] \cdot \frac{1}{s^{(k)} K} - 1 & \text{if } s^{(k)} > \frac{1}{K} \\
  z_t^{(k)} & \text{otherwise}
  \end{cases}
  \]
- **输出平衡后的教师概率分布**：
  \[
  p_t^{(k)} = \text{softmax}(z_t^{B(k)} / \tau_t)
  \]

##### **步骤5：对比损失优化**
- **损失函数**（交叉熵对比）：
  \[
  \mathcal{L} = \sum_{x \in \mathcal{G}} \left[ H(p_t(x), p_s^h(x)) + H(p_t(x), p_s^g(x)) \right]
  \]
  （\( H(a,b) = -a^T \log b \) 为交叉熵）

---

#### **3. 注意力预测模块集成**
- **输入**：GCN特征 \( \boldsymbol{v}_G \)，聚类分配 \( \boldsymbol{p}_t \)
- **注意力权重生成**：
  \[
  \alpha^{(k)} = \text{softmax}\left( \boldsymbol{v}_G \cdot \boldsymbol{c}_k \right) \quad (\boldsymbol{c}_k \text{为聚类中心})
  \]
- **最终预测**：
  \[
  \hat{y} = \sum_{k=1}^K \alpha^{(k)} \cdot \boldsymbol{c}_k
  \]

---

#### **4. 算法关键设计点**
1. **动态平衡约束**：
   - 通过 \( \mathcal{B} \) 强制聚类大小 \( s^{(k)} \approx \frac{1}{K} \)，避免少数聚类垄断样本。
   - 超参数 \( m_s \) 控制平衡灵敏度（高 \( m_s \) 稳定，低 \( m_s \) 灵敏）。

2. **无监督适应机制**：
   - 教师模型通过EMA更新（\( \theta_t \leftarrow m \theta_t + (1 - m) \theta_s \)），避免直接依赖数据分布。
   - 局部视图特征仅用于学生模型优化，教师模型仅用全局视图（防止噪声干扰）。

3. **计算效率**：
   - 硬分配统计仅需 \( O(N_B K) \) 计算量，无需存储全局特征。
   - 平衡操作符 \( \mathcal{B} \) 为逐元素操作，复杂度 \( O(K) \)。

---

#### **5. 适配TAD检测任务的调整建议**
- **特征增强策略**：
  - **全局视图**：保留Hi-C矩阵全结构，随机扰动节点顺序。
  - **局部视图**：截取局部TAD区域生成子图。
- **损失函数扩展**：
  - 联合优化目标：\( \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ExCB}} + \lambda_1 \mathcal{L}_{\text{edge}} + \lambda_2 \mathcal{L}_{\text{cur}} \)
  - \( \mathcal{L}_{\text{edge}} \): 边界特异性损失（检测Hi-C矩阵突变）
  - \( \mathcal{L}_{\text{cur}} \): 连续性损失（相邻区域特征相似性）

---

#### **总结**
ExCB的算法核心是通过动量统计和动态平衡操作符，在无监督条件下实现高效聚类分配。适配至GCN+Attention模型时，需将聚类平衡模块嵌入特征提取与注意力预测之间，通过以下流程：
1. **GCN提取特征** → 2. **ExCB伪聚类（动量更新 + \( \mathcal{B} \) 调整）** → 3. **注意力加权预测**  
此设计既保持ExCB的平衡性优势，又利用GCN捕捉图结构信息，最终提升下游任务（如TAD检测）的鲁棒性。