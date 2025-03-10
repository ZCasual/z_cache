图卷积网络（GCN）中的卷积操作通过聚合节点及其邻居的特征并应用线性变换和非线性激活实现。具体公式如下：

**公式推导与步骤：**
1. **添加自环**：将邻接矩阵 \( A \) 加上单位矩阵 \( I \)，得到 \( \tilde{A} = A + I \)，使每个节点在聚合时包含自身特征。
2. **计算度矩阵**：基于 \( \tilde{A} \) 计算度矩阵 \( \tilde{D} \)，其中 \( \tilde{D}_{ii} = \sum_j \tilde{A}_{ij} \)。
3. **对称归一化**：对邻接矩阵进行归一化处理，得到 \( \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \)，以平衡节点度的影响。
4. **特征聚合与变换**：将归一化的邻接矩阵与当前特征矩阵 \( H^{(l)} \) 相乘，聚合邻居信息，再通过可训练权重矩阵 \( W^{(l)} \) 进行线性变换。
5. **激活函数**：应用非线性激活函数 \( \sigma \)（如ReLU），得到下一层特征表示。

**最终公式：**
\[
H^{(l+1)} = \sigma\left( \hat{A} H^{(l)} W^{(l)} \right)
\]
其中：
- \( \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \) 为归一化后的邻接矩阵。
- \( \tilde{A} = A + I \) 是添加自环的邻接矩阵。
- \( \tilde{D} \) 是 \( \tilde{A} \) 的度矩阵。
- \( H^{(l)} \) 是第 \( l \) 层的节点特征矩阵。
- \( W^{(l)} \) 为可学习的权重矩阵。
- \( \sigma \) 为激活函数。

此公式通过结合局部邻居信息与非线性变换，有效捕获图结构数据中的复杂模式。