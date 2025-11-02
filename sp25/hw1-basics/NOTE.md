# CS336 HW1 源代码核心解析

## 整体架构

这是一个完整的 GPT 语言模型训练管道：

```
原始文本 → BPE分词器训练 → 文本编码为token序列 → 模型训练 → 文本生成
```

---

## 1. config.json - 配置文件

三个配置组：

**model**: 模型架构参数
- vocab_size=10000, context_length=256, d_model=512
- 4层Transformer, 16个注意力头, FFN维度1344
- RoPE的theta=10000

**optimizer**: 优化器参数
- 学习率从0.0002降到0.00001，500步warmup，10000步cosine退火
- weight_decay=0.01, 梯度裁剪=1.0

**train**: 训练参数
- batch_size=32, 训练6000步
- 每100步验证，每1000步保存检查点

---

## 2. utils.py - 设备工具

### `_to_device_and_compile(model)`

**功能**: 将模型移到最佳设备并编译加速

**逻辑**:
1. 检测设备优先级: MPS > CUDA > CPU
2. 移动模型到设备
3. 编译模型 (MPS用aot_eager后端，其他用默认)
4. 返回编译后的模型和设备

---

## 3. train_bpe.py - BPE分词器训练

### 主流程

**功能**: 训练一个10000词汇量的BPE分词器

**步骤**:
1. 调用 `run_train_bpe()` 在TinyStories训练集上训练
2. 输出vocab字典 (token_id → bytes) 和merges列表 (合并规则)
3. 用pickle序列化保存到磁盘
4. 打印最长token统计信息

**BPE原理**: 从256个字节token开始，迭代合并最频繁的token对，直到达到目标词汇量

---

## 4. tokenize.py - 数据预处理

### `encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path)`

**功能**: 将文本文件编码为token ID数组并保存为memmap

**逻辑**:
1. 第一遍遍历: 统计总token数
2. 创建np.memmap数组 (避免大数据集内存溢出)
3. 第二遍遍历: 逐行编码并写入memmap
4. flush到磁盘

**为什么用memmap**: 训练时可以随机访问任意位置的token，不需要全部加载到内存

### 主流程

1. 加载BPE分词器 (vocab + merges)
2. 测试编码/解码功能
3. 处理训练集和验证集，保存为train.dat和valid.dat

---

## 5. model.py - Transformer模型

### 核心组件

#### `softmax(x, dim=-1)`
标准softmax实现，先减去最大值防止溢出

#### `Linear(d_in, d_out)`
自定义线性层，使用truncated normal初始化 (fan-in fan-out)

#### `Embedding(vocab_size, d_model)`
词嵌入层，标准差=1的truncated normal初始化

#### `RMSNorm(hidden_size, eps=1e-5)`

**功能**: Root Mean Square归一化

**公式**: `output = x / sqrt(mean(x^2) + eps) * weight`

**关键点**: 
- 比LayerNorm少了mean centering，计算更快
- 先转fp32防止平方溢出，计算完转回原dtype

#### `RotaryEmbedding(context_length, dim, theta=10000)`

**功能**: RoPE位置编码

**初始化**:
- 预计算每个位置的cos/sin频率
- 频率公式: `theta^(-2i/dim)` 其中i是维度索引
- 缓存shape: [2, context_length, dim/2]

**forward(x, pos_ids)**:
- 将输入分成相邻的对 (x1, x2)
- 对每对应用2D旋转: `[cos*x1-sin*x2, sin*x1+cos*x2]`
- 拼接回原shape

**优势**: 相对位置编码，支持长度外推

#### `scaled_dot_product_attention(Q, K, V, mask=None)`

**功能**: 缩放点积注意力

**公式**: `softmax(Q·K^T / sqrt(d_k)) · V`

**步骤**:
1. 计算注意力分数: `Q @ K.T / sqrt(d_k)`
2. 应用mask (因果mask将未来位置设为-inf)
3. softmax归一化
4. 加权求和V

#### `CausalMultiHeadSelfAttention(d_model, num_heads, positional_encoder)`

**功能**: 多头因果自注意力

**forward(x, token_positions=None)**:
1. 线性投影得到Q, K, V
2. reshape为多头: `[batch, seq, d_model] → [batch, heads, seq, d_k]`
3. 对Q, K应用RoPE位置编码
4. 构造因果mask: `qi >= kj` (位置i只能看到≤i的位置)
5. 调用scaled_dot_product_attention
6. concat多头输出并投影回d_model

#### `SwiGLU(d_model, d_ff)`

**功能**: 门控前馈网络

**公式**: `w2(silu(w1(x)) * w3(x))`

**对比标准FFN**: 
- 标准: `w2(relu(w1(x)))`
- SwiGLU: 用两个门的乘积，性能更好

#### `TransformerBlock(d_model, num_heads, d_ff, positional_encoder)`

**功能**: 单个Transformer层

**forward(x)**:
```
x = x + attn(norm(x))      # 注意力子层
x = x + ffn(norm(x))       # 前馈子层
```

**Pre-Norm架构**: 在子层之前做归一化，训练更稳定

#### `BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)`

**功能**: 完整的语言模型

**结构**:
```
token_ids → Embedding → [TransformerBlock × num_layers] → RMSNorm → Linear → logits
```

**forward(x)**: 
- 输入token IDs，输出每个位置的next-token logits

**generate(x, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None)**:

自回归生成，每步:
1. 如果序列超过context_length，裁剪到最后context_length个token
2. 前向得到logits，取最后一个位置
3. 温度缩放: `logits /= temperature`
4. top-k过滤: 只保留概率最高的k个token
5. softmax + 多项式采样
6. 拼接到序列，遇到EOS或达到max_new_tokens停止

**from_pretrained(path)**: 从磁盘加载模型配置和权重

---

## 6. train.py - 模型训练

### 数据加载函数

#### `get_memmap_dataset(path, dtype=np.int32)`
打开memmap文件为只读数组

#### `get_batch(memmap_arr, batch_size, context_length)`

**功能**: 随机采样训练batch

**逻辑**:
1. 随机选batch_size个起始位置
2. 每个位置取连续context_length个token作为x
3. 对应的y是x向右偏移1位 (next token prediction)

#### `memmap_val_iterator(memmap_arr, batch_size, context_length)`

**功能**: 顺序遍历验证集

**逻辑**: 将数据集分成不重叠的batch，逐个yield

### 主训练流程

1. **初始化**:
   - 加载config，构建模型
   - 移到设备并编译
   - 加载训练/验证数据
   - 创建AdamW优化器

2. **恢复断点** (如果指定):
   - 加载checkpoint
   - 恢复模型权重、优化器状态、迭代数

3. **训练循环** (每步):
   - 随机取batch
   - 前向计算logits
   - 计算cross entropy loss
   - 反向传播
   - 梯度裁剪 (防止梯度爆炸)
   - 更新学习率 (cosine schedule with warmup)
   - 优化器step

4. **定期验证**:
   - 每val_interval步
   - 在val_batches个batch上计算平均loss

5. **定期保存**:
   - 每save_interval步
   - 保存模型权重、优化器状态、迭代数

**学习率调度**: 
- 前warmup_iters步线性增长到lr
- 之后cosine退火到min_lr

---

## 7. generate.py - 文本生成

### 主流程

1. **加载分词器**: 读取vocab和merges，构建Tokenizer
2. **加载模型**: 
   - 从config构建模型结构
   - 从checkpoint加载权重
   - 移到设备并编译
3. **编码prompt**: 文本 → token IDs
4. **生成**: 调用model.generate()
5. **解码**: token IDs → 文本
6. **输出**: 打印输入和完整生成结果

**命令行参数**:
- `--prompt`: 输入提示文本
- `--max_new_tokens`: 最多生成多少个token
- `--temperature`: 控制随机性 (越小越确定)
- `--top_k`: 只从概率最高的k个token中采样

---

## 核心设计思想

### 数据流

1. **训练前**: 原始文本 → BPE训练 → 得到vocab/merges → 编码所有文本为memmap
2. **训练时**: memmap随机采样 → 模型前向 → loss反向 → 更新参数
3. **推理时**: prompt编码 → 自回归生成 → 解码输出

### 效率优化

- **memmap**: 大数据集不占内存
- **torch.compile**: 加速模型执行
- **Pre-Norm**: 训练稳定，收敛快
- **梯度裁剪**: 防止训练崩溃

### 现代化设计

- **RoPE**: 相对位置编码，支持长度外推
- **SwiGLU**: 比ReLU效果好
- **RMSNorm**: 比LayerNorm快
- **Cosine schedule**: 学习率平滑衰减

### 模块化

每个文件职责清晰:
- config: 超参数
- utils: 通用工具
- train_bpe: 分词器
- tokenize: 数据预处理
- model: 模型定义
- train: 训练逻辑
- generate: 推理逻辑

便于调试、扩展和复用
