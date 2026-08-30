# 运行和理解 Llama 2 C/C++ 推理 Runtime

本文说明如何使用仓库中的四个 CPU 推理程序，以及一个 token 如何经过分词、Transformer、采样和解码成为输出文本。内容以当前源码、构建文件和测试为准，适合首次运行示例的读者，也适合需要核对 checkpoint、模块边界和 C/C++ 实现差异的开发者。

## 1. 概述

### 1.1 选择 Runtime

仓库提供 FP32 和 Q8_0 两条推理路径。每条路径都有 C 和 C++20 实现：

| 程序 | 语言 | Checkpoint | 主要特点 |
| --- | --- | --- | --- |
| [`run.c`](../run.c) | C | legacy v0 FP32 | 上游参考实现，使用裸指针和显式释放 |
| [`run.cpp`](../run.cpp) | C++20 | legacy v0 FP32 | 使用 RAII、`vector` 和 `span` 管理所有权与长度 |
| [`runq.c`](../runq.c) | C | v2 Q8_0 | 量化大矩阵和线性层输入 |
| [`runq.cpp`](../runq.cpp) | C++20 | v2 Q8_0 | 增加量化视图、边界检查和显式 group size |

四个程序都在 CPU 上逐 token 推理。一个进程加载一个模型，并维护一份会话状态和 KV Cache。它们不提供 batch、请求调度、GPU kernel、分布式执行或网络服务。

> [!IMPORTANT]
> `run`/`run_cpp` 与 `runq`/`runq_cpp` 不能交换 checkpoint。FP32 Runtime 读取 v0，Q8_0 Runtime 读取 v2。`export.py` 还能写 v1，但当前四个 Runtime 都不能读取 v1。

### 1.2 快速运行

下面的命令构建 C++20 FP32 Runtime，并让 15M 参数 TinyStories 模型续写文本。开始前需要 C++20 编译器、`make`、`curl`，以及仓库根目录中的 `tokenizer.bin`。

在仓库根目录下载 v0 FP32 checkpoint：

```bash
curl -L \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin \
  -o stories15M.bin
```

构建 `run_cpp`：

```bash
make runcpp
```

使用 greedy sampling 生成文本：

```bash
./run_cpp stories15M.bin \
  -z tokenizer.bin \
  -i "Once upon a time" \
  -t 0 \
  -n 32
```

标准输出包含 prompt 和续写。标准错误在推理结束后输出近似吞吐量：

```text
achieved tok/s: ...
```

> [!NOTE]
> `-n` 指定本次进程最多处理的总 position 数，其中包含 prompt token。传入 `0` 或大于模型 `seq_len` 的值时，Runtime 使用 `seq_len`。

### 1.3 完整数据路径

一次生成由四个主要组件完成：

```text
CLI
  -> Transformer 加载 checkpoint
  -> Tokenizer 加载词表
  -> Sampler 保存采样参数和随机状态
  -> generate() 或 chat()
```

文本随后沿以下路径循环：

```text
prompt 文本
  -> Tokenizer::encode()
  -> token IDs
  -> Transformer::forward(token, position)
  -> logits
  -> Sampler::sample()
  -> next token ID
  -> Tokenizer::decode()
  -> 输出文本
  -> 下一次 forward()
```

Prompt 尚未消费完时，`generate()` 强制使用 prompt 中的下一个 token，不采纳模型预测。每个 prompt token 仍会执行完整 forward，并把当前层的 Key 和 Value 写入 KV Cache。Prompt 结束后，采样结果才成为下一轮输入。

### 1.4 CLI 参数

四个程序接受相同的参数集合，可执行文件名和 checkpoint 版本不同：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| 第一个位置参数 | 必填 | checkpoint 路径 |
| `-z PATH` | `tokenizer.bin` | tokenizer binary 路径 |
| `-i TEXT` | 空 | 初始 prompt |
| `-n N` | `256` | 总 position 上限 |
| `-t T` | `1.0` | temperature；`0` 使用 argmax |
| `-p P` | `0.9` | top-p；`0` 或 `1` 使用完整分布 |
| `-s SEED` | 自动生成 | 正整数固定随机种子 |
| `-m MODE` | `generate` | 选择 `generate` 或 `chat` |
| `-y TEXT` | 空 | 第一轮 Chat 的 system prompt |

C 使用 `atoi()` 和 `atof()` 解析数字。C++ 使用 `from_chars()` 和 `strtof()`，并拒绝尾随字符、溢出和非有限浮点数。两者都会把负 temperature 改为 0，把范围外的 top-p 改为 0.9。

## 2. 工作原理

### 2.1 Transformer 层和 Attention head

`n_layers` 和 `n_heads` 描述模型的不同层级：

| 配置 | 含义 |
| --- | --- |
| `n_layers` | 串行执行的 Transformer 层数 |
| `n_heads` | 每层 Attention 中的 Query head 数量 |

每个 Transformer 层按以下顺序处理输入：

```text
输入 x
  -> Attention RMSNorm
  -> 一次多头 Attention
  -> Attention 残差相加
  -> MLP RMSNorm
  -> 一次 MLP / FFN
  -> MLP 残差相加
  -> 本层输出
```

整个模型按层执行：

```text
Embedding
  -> Transformer 层 0：多头 Attention 0 -> MLP 0
  -> Transformer 层 1：多头 Attention 1 -> MLP 1
  -> Transformer 层 2：多头 Attention 2 -> MLP 2
  -> ...
  -> Final RMSNorm
  -> Wcls
  -> logits
```

> [!IMPORTANT]
> 模型不会先执行所有 Attention，再执行所有 MLP。每层先完成自己的 Attention 和 MLP，然后把结果传给下一层。各层使用各自的训练权重。

Llama 2 使用 Pre-Norm 和残差连接：

$$
h=x+\operatorname{Attention}(\operatorname{RMSNorm}(x))
$$

$$
y=h+\operatorname{MLP}(\operatorname{RMSNorm}(h))
$$

当前层的 `y` 成为下一层的 `x`。

每个 Transformer 层包含一个多头 Attention 模块。该模块内部包含多个 Query head。例如，`n_layers = 6`、`n_heads = 8` 表示：

- 模型串行执行 6 个 Transformer 层；
- 每层包含一个多头 Attention；
- 每个多头 Attention 包含 8 个 Query head；
- 每层拥有独立的 Wq、Wk、Wv、Wo 和 MLP 权重。

Checkpoint 可以先连续存储所有层的 Wq，再存储所有层的 Wk，最后存储 MLP 权重。该顺序只定义权重在文件中的位置，不改变运行时顺序：

```text
第 0 层 Attention -> 第 0 层 MLP
第 1 层 Attention -> 第 1 层 MLP
...
```

### 2.2 模型配置

`Config` 包含七个架构整数。它们决定权重形状、临时缓冲区和 KV Cache 大小：

| 字段 | 含义 |
| --- | --- |
| `dim` | residual stream 和 token 表征宽度 |
| `hidden_dim` | MLP 中间宽度 |
| `n_layers` | Transformer 层数 |
| `n_heads` | 每层 Query head 数量 |
| `n_kv_heads` | 每层 Key/Value head 数量 |
| `vocab_size` | 词表大小和 logits 数量 |
| `seq_len` | 最大 position 数 |

Runtime 使用以下派生维度：

$$
\mathrm{head\_size}=\frac{\mathrm{dim}}{\mathrm{n\_heads}}
$$

$$
\mathrm{kv\_dim}=\mathrm{head\_size}\times\mathrm{n\_kv\_heads}
$$

C++ 的 `Config::validate()` 要求所有字段为正，`dim` 可被 `n_heads` 整除，`n_heads` 可被 `n_kv_heads` 整除，并且 `head_size` 为偶数。最后一个条件用于 RoPE 的两两坐标旋转。C 实现没有等价的集中式校验。

`n_kv_heads` 决定 Query head 如何共享 Key 和 Value：

- `n_kv_heads == n_heads` 表示 Multi-Head Attention（MHA）；
- `1 < n_kv_heads < n_heads` 表示 Grouped-Query Attention（GQA）；
- `n_kv_heads == 1` 表示 Multi-Query Attention（MQA）。

Runtime 不复制共享的 KV head。它使用整数映射：

```text
queries_per_kv_head = n_heads / n_kv_heads
kv_head = query_head / queries_per_kv_head
```

这些数值属于训练好的模型结构。推理时不能在不更换权重的情况下修改它们。

### 2.3 `forward()`

`forward(token, position)` 接收一个 token ID 和一个 position，返回长度为 `vocab_size` 的 logits。Logit 是一个候选 token 的未归一化分数，不是概率。只有随机采样路径才会把 logits 变为概率。

#### Token embedding

FP32 Runtime 从 embedding 表复制当前 token 对应的一行：

$$
x=E[\mathrm{token},:]
$$

Q8 C++ Runtime 定位量化 embedding 行，并只把这一行反量化到 FP32 `x`。`runq.c` 则在启动时反量化完整 embedding 表。

#### RMSNorm

Attention 和 MLP 之前各执行一次 RMSNorm：

$$
y_i=w_i\frac{x_i}{\sqrt{\frac{1}{N}\sum_j x_j^2+10^{-5}}}
$$

RMSNorm 不减去均值。四个 Runtime 都把 epsilon 固定为 `1e-5`。

#### Query、Key 和 Value

归一化后的表征进入三组不同的训练权重：

$$
q=W_q\hat{x},\qquad k=W_k\hat{x},\qquad v=W_v\hat{x}
$$

Value 由 `Wv` 直接投影得到，不是由 score 或 Softmax 计算得到。Query 只供当前 position 使用。Key 和 Value 会被后续 position 再次读取，因此写入当前层的 KV Cache。

#### RoPE

RoPE 对 Query 和 Key 的相邻坐标执行二维旋转：

$$
a'=a\cos\theta-b\sin\theta
$$

$$
b'=a\sin\theta+b\cos\theta
$$

Runtime 不旋转 Value。KV Cache 保存已经旋转的 Key。当前实现把 RoPE 基数固定为 10000，并在 forward 时计算正弦和余弦。

#### 多头 Attention

每个 Query head 与 position 0 到当前位置的 Key 做缩放点积：

$$
\mathrm{score}_t=\frac{q\cdot k_t}{\sqrt{\mathrm{head\_size}}}
$$

Runtime 对这一行 score 执行稳定 Softmax：

$$
p_t=\frac{e^{\mathrm{score}_t-m}}{\sum_j e^{\mathrm{score}_j-m}}
$$

其中 `m` 是这一行的最大 score。当前 head 的输出是历史 Value 的加权和：

$$
o=\sum_{t=0}^{\mathrm{position}}p_t v_t
$$

代码只读取当前位置及其之前的缓存，因此自然满足 causal 条件。所有 head 的输出写入 `xb` 中不重叠的片段。Wo 把拼接结果投影回 `dim`，并通过残差连接加入 `x`：

$$
x\leftarrow x+W_o\operatorname{Concat}(o_0,o_1,\ldots)
$$

#### SwiGLU MLP

本仓库中的 MLP 和 FFN 指同一个 Feed-Forward Network 子层。它使用两次输入投影：

$$
g=W_1\hat{x},\qquad u=W_3\hat{x}
$$

随后执行 SiLU 和逐元素门控：

$$
h=\operatorname{SiLU}(g)\odot u
$$

W2 把 `hidden_dim` 投影回 `dim`，并完成第二次残差相加：

$$
x\leftarrow x+W_2h
$$

#### Final RMSNorm 和 Wcls

所有层完成后，Runtime 执行 Final RMSNorm，再使用 language-model head 生成 logits：

$$
\mathrm{logits}=W_{\mathrm{cls}}\operatorname{RMSNorm}(x)
$$

Wcls 不是 Softmax。若 checkpoint 使用 weight tying，Wcls 与 token embedding 指向同一组权重；否则 checkpoint 单独保存 Wcls。

### 2.4 生成循环

`generate()` 从 BOS token 开始，逐个处理 prompt token，然后进入自回归 decode：

```text
当前 token + position
  -> forward()
  -> logits
  -> prompt 中还有 token？
       是：使用下一个 prompt token
       否：Sampler 选择 next token
  -> decode() 并输出
  -> next token 成为下一轮输入
```

当前实现没有矩阵化 prefill kernel。Prompt processing 和 decode 都一次处理一个 token。`generate()` 遇到 token ID 1（BOS）时结束，这沿用上游控制逻辑。

`chat()` 使用 Llama 2 instruction 模板。第一轮可以包含 system prompt；后续 user prompt 从标准输入读取。Position 在整个进程内递增，一份 KV Cache 保存当前会话的全部历史。C++ Chat 会把 EOS token 也送入 forward，再开始读取下一轮输入。

### 2.5 模块和内存

C++ Runtime 把模型分成四类对象：

- `MappedFile` 以只读、私有 `mmap` 映射 checkpoint，并在析构时 `munmap()`；
- `TransformerWeights` 保存指向映射数据的非拥有视图；
- `RunState` 拥有 forward 会修改的 activation、scratch 和 KV Cache；
- `Transformer` 组合配置、权重和运行状态，并提供 `forward()`。

`MappedFile` 在权重视图之前构造，在它们之后析构。这样所有 `span` 失效后，底层映射才被解除。映射通常按需调页，不表示构造时复制整个 checkpoint。

`WeightCursor` 按格式顺序切分权重。FP32 版本以 float 为单位推进；Q8 版本以 byte 为单位推进，因为 payload 同时包含 int8 values 和 FP32 scales。C++ cursor 会在每次切分前检查剩余长度，但不会取得数据所有权。

`RunState` 在构造时分配以下主要缓冲区：

| 缓冲区 | 大小 | 用途 |
| --- | ---: | --- |
| `x`、`xb`、`xb2` | 各 `dim` | residual stream 和层内临时结果 |
| `hb`、`hb2` | 各 `hidden_dim` | SwiGLU 两个分支 |
| `q` | `dim` | 当前 token 的 Query heads |
| `attention` | `n_heads * seq_len` | score 和 Softmax 概率的复用空间 |
| `logits` | `vocab_size` | Wcls 输出 |
| `key_cache`、`value_cache` | 各 `n_layers * seq_len * kv_dim` | 每层、每个 position 的 K/V |

Q8 版本还拥有 `xq` 和 `hq`，分别保存模型宽度与 MLP 中间 activation 的 int8 values 和 FP32 scales。Forward 的正常路径会复用这些缓冲区。

### 2.6 FP32 和 Q8_0

FP32 与 Q8_0 Runtime 执行相同的模型算法。差异位于大矩阵的存储形式和线性层输入。

Q8_0 把每组连续 `G` 个 FP32 值映射到 int8：

$$
s_g=\frac{\max_{i\in g}|x_i|}{127}
$$

$$
q_i=\operatorname{clamp}(\operatorname{round}(x_i/s_g),-127,127)
$$

反量化近似为：

$$
x_i\approx q_i s_g
$$

Q8 矩阵乘先对每组执行 int8 点积，再乘矩阵 scale 和输入 scale：

$$
y_r=\sum_g\left(\sum_{k\in g}q^W_{r,k}q^x_k\right)s^W_{r,g}s^x_g
$$

C++ activation 量化会把全零组写成零 scale 和全零 values，避免除以零。组内点积使用 `int32_t`，跨组累加结果使用 FP32。

Q8 C++ forward 中的量化位置如下：

```text
Q8 embedding 行 -> 反量化 -> FP32 x

Attention RMSNorm -> 量化 xq
  -> Wq / Wk / Wv Q8 matmul
  -> FP32 RoPE、Attention、KV Cache
  -> 量化 xq -> Wo Q8 matmul -> FP32 残差

MLP RMSNorm -> 量化 xq
  -> W1 / W3 Q8 matmul
  -> FP32 SwiGLU
  -> 量化 hq -> W2 Q8 matmul -> FP32 残差

Final RMSNorm -> 量化 xq -> Wcls Q8 matmul -> FP32 logits
```

> [!IMPORTANT]
> “Q8 Runtime”不表示整个 forward 都使用 int8。RMSNorm、RoPE、Attention score、Softmax、Value 加权、KV Cache、残差流、SwiGLU 输出和 logits 仍使用 FP32。

若 `G = 64`，一个 group 从 256 bytes 的 FP32 数据变为 64 bytes int8 values 加 4 bytes scale，单看该 group 的存储约缩小 3.76 倍。这个比例不包括 FP32 norm、header 和运行状态，也不保证 Q8 在所有 CPU 上更快。

### 2.7 Checkpoint

#### v0 FP32

V0 没有 magic、version 或固定长度 header。文件先保存 28-byte `Config`，再按以下顺序保存 FP32 数据：

```text
token embedding
所有层 Attention RMSNorm
所有层 Wq
所有层 Wk
所有层 Wv
所有层 Wo
所有层 MLP RMSNorm
所有层 W1
所有层 W2
所有层 W3
Final RMSNorm
legacy RoPE cosine table
legacy RoPE sine table
可选 Wcls
```

`Config.vocab_size` 的符号编码分类器是否共享。正值表示 Wcls 与 embedding 共享；负值表示文件末尾包含独立 Wcls。Runtime 取绝对值得到实际词表大小。

当前 forward 不读取两张 legacy RoPE 表，而是动态计算旋转。Reader 仍必须跳过这两块数据，才能找到可选 Wcls。

#### v1 FP32

V1 使用 256-byte header。Offset 0 保存 magic `0x616b3432`，offset 4 保存 version 1，offset 8 保存七个配置整数，offset 36 保存 `shared_classifier`。Payload 先保存 norm，然后保存 embedding、每类层权重和可选 Wcls，全部使用 FP32。

`export.py` 实现了 v1 writer，但仓库没有 v1 reader。

#### v2 Q8_0

V2 同样使用 256-byte header。它保存 magic、version 2、`Config`、`shared_classifier` 和 `group_size`。Group size 从 offset 37 开始，因此 C++ reader 使用 `memcpy()` 读取，避免未对齐解引用。

Payload 先保存 FP32 norm：

```text
所有层 Attention RMSNorm
所有层 MLP RMSNorm
Final RMSNorm
```

随后保存 embedding、Wq、Wk、Wv、Wo、W1、W2、W3 和可选 Wcls。每个量化 tensor 独立保存：

```text
int8 values[element_count]
FP32 scales[element_count / group_size]
```

它不是先保存全模型所有 values，再保存全模型所有 scales。

### 2.8 Tokenizer

Tokenizer binary 不保存词表大小。Runtime 从 checkpoint 的 `Config.vocab_size` 决定读取多少项。文件布局为：

```text
native uint32 max_token_length
重复 vocab_size 次：
  native FP32 score
  native uint32 piece_length
  byte[piece_length] piece
```

`encode()` 按以下顺序处理文本：

1. 按调用要求加入 BOS token，ID 为 1。
2. 非空文本前加入 SentencePiece dummy-prefix 空格。
3. 按 UTF-8 code point 查找完整 piece。
4. 找不到完整 piece 时，把原始字节映射为 `byte + 3`。
5. 反复选择词表分数最高的相邻 pair，并合并为一个 token。
6. 按调用要求加入 EOS token，ID 为 2。

`decode()` 从 token ID 取得 piece，移除 BOS 后的 dummy prefix，并把 `<0x41>` 一类 byte token 恢复为原始字节。输出函数会过滤不可打印的单字节控制字符。

### 2.9 Sampler

Sampler 根据 temperature 和 top-p 选择一条路径：

- `temperature == 0`：直接选择最大 logit 的 token；
- `temperature != 0` 且 top-p 为 0 或 1：缩放 logits、执行 Softmax，再从完整分布采样；
- `temperature != 0` 且 `0 < top_p < 1`：缩放 logits、执行 Softmax，再执行 top-p sampling。

Top-p sampling 按概率从大到小排列候选，找到累计概率首次超过阈值的最小前缀，再在该前缀内采样。实现会先用安全 cutoff 排除不可能进入 nucleus 的低概率 token，以减少排序数量。

C 和 C++ 随后都使用 xorshift64* 生成随机数，但自动 seed 来源不同。C 使用秒级 `time(NULL)`；C++ 使用 `random_device`，并确保内部状态非零。

### 2.10 C 与 C++ 映射

C++ 重构保留 C Runtime 的模型算法，并把资源生命周期和长度信息放入类型中：

| C | C++ | 变化 |
| --- | --- | --- |
| `malloc_run_state` / `free_run_state` | `RunState` | vector 自动释放 |
| checkpoint mmap 字段 | `MappedFile` | RAII 管理映射 |
| `memory_map_weights` | `WeightCursor` | 裸指针推进改为有边界切片 |
| `float*` 权重 | `span<const float>` | 非拥有视图携带长度 |
| C Q8 `QuantizedTensor` | C++ `QuantizedTensor` | values 和 scales 都携带长度 |
| 手动分配 `xq` / `hq` | `QuantizedBuffer` | 类型表达可写所有权 |
| 全局 `GS` | `Transformer::group_size_` | group size 属于模型实例 |
| `rmsnorm` / `matmul` / `softmax` | `kernel` 函数 | 增加输入大小和形状检查 |
| `build_tokenizer` | `Tokenizer` 构造函数 | 文件和词表自动清理 |
| `build_sampler` | `Sampler` 构造函数 | 保存配置、随机状态和排序 scratch |
| `forward` | `Transformer::forward` | 配置、权重和状态成为类成员 |

C++ 改善了所有权、异常路径和多数边界检查。它没有新增 GPU、服务或分布式能力。C++ kernel 仍是单线程标量循环；C 源码中的 OpenMP pragma 只有通过 `make runomp` 构建时才生效。

### 2.11 复杂度

设 `D = dim`、`F = hidden_dim`、`K = kv_dim`、`L = n_layers`、`T` 为当前可见历史长度、`V = vocab_size`。

每层、每个 decode token 的主要乘加量来自：

```text
Wq + Wo                  约 2D²
Wk + Wv                  约 2DK
W1 + W3 + W2             约 3DF
score + Value 加权        约 2TD
```

最终 Wcls 还需要约 `VD` 次乘加。因此总体量级为：

$$
O\left(L(D^2+DK+DF+TD)+VD\right)
$$

Attention 的历史读取随 `T` 线性增长。因为当前 prefill 仍逐 token 执行，长度为 `N` 的 prompt 在 Attention 部分累计为二次量级。

除映射权重外，最大的持久状态通常是 KV Cache：

$$
\mathrm{KV\ bytes}=2\times L\times\mathrm{seq\_len}\times K\times4
$$

末尾的 4 表示 FP32 字节数。Q8 Runtime 也使用 FP32 KV Cache。GQA 和 MQA 通过减小 `K` 同时减少 Wk/Wv 参数和缓存。

Q8 tensor 中每个量化值的平均存储为：

$$
1+\frac{4}{G}\ \mathrm{bytes}
$$

该估算不包括 FP32 norm、header 和运行状态。

## 3. 示例

### 3.1 构建程序

构建 C FP32 和 Q8_0 程序：

```bash
make run
```

构建两个 C++20 程序：

```bash
make runcpp
make runqcpp
```

Makefile 还提供以下目标：

| Target | 结果 |
| --- | --- |
| `rundebug` | C FP32/Q8，`-g` |
| `runcppdebug` | C++ FP32，`-O0 -g` |
| `runfast` | C FP32/Q8，`-Ofast` |
| `runcppfast` | C++ FP32，`-O3 -ffast-math` |
| `runomp` | C FP32/Q8，OpenMP 和 `-march=native` |
| `win64` | MinGW C `run.exe` 和 `runq.exe` |
| `rungnu` / `runompgnu` | GNU11 C 构建 |
| `clean` | 删除已知可执行文件和 C++ dSYM 目录 |

直接执行 `make` 会选择 Makefile 的第一个 target `run`，因此只构建两个 C 程序。当前没有 C++ Q8 debug/fast target、C++ OpenMP target 或 C++ Windows target。

### 3.2 运行生成

Greedy sampling 适合可重复的短检查：

```bash
./run_cpp stories15M.bin \
  -z tokenizer.bin \
  -i "Once upon a time" \
  -t 0 \
  -n 128
```

随机采样需要显式设置 seed：

```bash
./run_cpp stories15M.bin \
  -z tokenizer.bin \
  -i "Once upon a time" \
  -t 1.0 \
  -p 0.9 \
  -s 42 \
  -n 128
```

比较 C 与 C++ 的一次确定性输出：

```bash
./run stories15M.bin -z tokenizer.bin -i "Once" -t 0 -n 8 \
  > /tmp/llama2-c.txt
./run_cpp stories15M.bin -z tokenizer.bin -i "Once" -t 0 -n 8 \
  > /tmp/llama2-cpp.txt
cmp /tmp/llama2-c.txt /tmp/llama2-cpp.txt
```

`cmp` 不输出内容且返回 0，表示这组输入的标准输出相同。它是一次点验证，不表示所有模型和边界条件都已覆盖。

### 3.3 运行 Chat

Chat 模式需要与 Llama 2 instruction 模板匹配的 instruct/chat checkpoint：

```bash
./run_cpp llama2_7b_chat.bin \
  -z tokenizer.bin \
  -m chat \
  -y "You are a concise assistant." \
  -i "Explain KV Cache."
```

第一轮使用 `-y` 和 `-i`。后续轮次从标准输入读取。Base 模型或 TinyStories 模型不保证理解 Chat 模板。

> [!WARNING]
> C 版 `chat()` 使用固定长度缓冲区以及无边界的 `strcpy()` 和 `sprintf()`。过长的 CLI prompt 或拼接结果可能越界。不要把 C Chat 路径用于不可信输入。C++ 版改用动态 string 和 vector，但仍不是多用户服务。

### 3.4 导出 checkpoint

从本仓库训练得到的 PyTorch checkpoint 导出 v0 FP32：

```bash
python export.py model.bin \
  --version 0 \
  --checkpoint out/ckpt.pt
./run_cpp model.bin -z tokenizer.bin -t 0 -n 32
```

导出并运行 v2 Q8_0：

```bash
python export.py model_q80.bin \
  --version 2 \
  --checkpoint out/ckpt.pt
make runqcpp
./runq_cpp model_q80.bin -z tokenizer.bin -t 0 -n 32
```

默认 group size 为 64。导出器会在 `dim` 不能整除 group size 时反复减半。所有量化 tensor 的元素数仍必须可整除最终 group size；C++ Q8 loader 还明确要求 `dim` 和 `hidden_dim` 都可整除它。

生成 v1 文件：

```bash
python export.py model_v1.bin \
  --version 1 \
  --checkpoint out/ckpt.pt
```

> [!IMPORTANT]
> 上一条命令只生成文件。当前仓库没有 v1 reader，因此不能用 `run`、`run_cpp`、`runq` 或 `runq_cpp` 执行它。

导出器接受三种互斥输入：

```bash
python export.py model.bin --version 0 --checkpoint out/ckpt.pt
python export.py model.bin --version 0 --meta-llama /path/to/meta/model
python export.py model.bin --version 0 --hf /path/to/hf/model
```

运行 `python export.py --help` 可查看当前参数。使用 Meta 或 Hugging Face 输入前，请先检查 [`export.py` 的已知问题](#43-exportpy-的已知问题)。

### 3.5 运行测试

默认 pytest 需要 Python 依赖和 C 程序：

```bash
python -m pip install -r requirements.txt
make run
pytest
```

`pytest` 可能联网下载 stories260K 测试文件。它包含两项测试：

- `test_runc` 运行 `run.c` 200 steps，并与固定输出逐字节比较；
- `test_python` 使用 PyTorch 模型生成 200 tokens，并与同一固定输出比较。

只运行名称匹配 `runc` 的 pytest：

```bash
make testc
```

单独编译并运行 C tokenizer 的五组编码测试：

```bash
make testcc
```

增加 tokenizer 测试输出时使用 `make testcc VERBOSITY=1`。`test.c` 不属于默认 pytest。

## 4. 重要提示

### 4.1 Checkpoint 兼容性

文件扩展名不能证明 checkpoint 版本。V2 可以通过 magic 和 version 识别；v0 没有自描述 header。调用方应明确管理格式：

- `run` 和 `run_cpp` 把文件开头直接解释为 v0 `Config`；
- `runq` 和 `runq_cpp` 明确要求 v2；
- v1 当前只有 writer，没有 reader。

> [!WARNING]
> 不要把当前 loader 当作面向敌对文件的安全解析器。格式没有 checksum、hash、签名或逐 tensor 长度；C loader 缺少逐区域截断检查，C++ loader 也没有为所有大小乘法统一检查溢出。

C++ cursor 会发现多数过短 payload，但四个 loader 都不拒绝尾随字节。同长度的内容损坏也可能被当作合法权重继续执行。从外部获取 checkpoint 时，应另行验证可信 hash。

`export.py` 的 `struct.pack()` 没有指定固定端序，Reader 也按本机整数和 float 表示读取。Checkpoint 和 tokenizer 实际依赖 native endian、32-bit 整数和 IEEE-754 FP32。常见 little-endian x86-64/ARM64 通常可以互换，但格式没有定义 big-endian 兼容路径。

### 4.2 模型和 Tokenizer 兼容性

四个 Runtime 把以下值写死在源码中：

```text
RMSNorm epsilon = 1e-5
RoPE theta       = 10000
```

V0、v1 和 v2 `Config` 都不保存这些值。V0 虽然带两张 RoPE 表，Runtime 仍跳过它们并按 10000 动态计算。若源模型使用不同 `norm_eps`、`rope_theta`、RoPE scaling 或其他位置编码，文件能导出并不表示推理语义正确。

Tokenizer 还假设：

- BOS ID 为 1；
- EOS ID 为 2；
- byte fallback 从 ID 3 开始；
- 非空输入使用 SentencePiece dummy prefix；
- tokenizer 词表项数和 checkpoint `vocab_size` 完全一致。

它不是通用 Hugging Face tokenizer Runtime。不同的 special token、normalization、added token 或 pre-tokenization 规则会改变模型输入。

### 4.3 `export.py` 的已知问题

当前导出器具有以下边界：

1. `--version 1` 只写文件，仓库没有对应 reader。
2. V0/v1 固定写 FP32；v2 固定写 Q8_0 大矩阵和 FP32 norm。`--dtype` 不会改变这些格式。
3. Hugging Face 导入把 `n_kv_heads` 设置为 `num_attention_heads`，没有读取 `num_key_value_heads`。GQA/MQA 模型会被错误地当成 MHA。
4. Hugging Face Wk 的逆 permutation 假设完整 `dim * dim` 形状，没有实现缩窄 KV 投影的完整路径。
5. Meta 导入把 `max_seq_len` 固定为 2048。
6. Runtime checkpoint 不保存源模型的 `norm_eps`、RoPE theta 或 RoPE scaling。
7. V2 group-size backoff 只直接检查 `dim`。其他 tensor 不整除时，后续 assertion 才会失败。
8. 该 assertion 的错误消息引用未定义的循环变量 `i`，触发时可能得到 `NameError`，而不是预期说明。
9. V2 权重量化没有显式处理全零 group；零 scale 会进入除零路径。C++ Runtime 的 activation 量化已经处理该情况。
10. CLI 虽解析了 `--dtype`，却把原始字符串作为第三个位置参数传给 `model_export()`。在 `--version -1` 路径中，该字符串又落到 `hf_export()` 的 `group_size` 参数，实际导出仍使用默认 FP32 dtype。

> [!IMPORTANT]
> “`AutoModelForCausalLM` 能加载”不表示任意 Hugging Face Llama 变体都能正确转换。特别是 GQA/MQA、不同 RoPE 配置和不同 tokenizer 规则，需要先修正导入与格式字段。

### 4.4 执行和平台限制

当前 Runtime 不实现：

- GPU、CUDA、Metal 或其他加速器 kernel；
- FlashAttention、Online Softmax、BLAS 或显式 SIMD；
- batch、continuous batching、动态请求调度或 paged KV Cache；
- prefix cache、量化 KV Cache 或 speculative decoding；
- Tensor、Pipeline、Data Parallel 或分布式通信；
- HTTP/gRPC、鉴权、限流、租户隔离或服务可观测性；
- 同一 `Transformer` 实例的并发安全保证。

这些程序适合阅读和验证逐 token CPU 执行链，不能直接替代生产 serving engine。

POSIX C/C++ 路径依赖 `open`、`fstat`、`mmap` 和 `munmap`。Windows 兼容函数位于 [`win.h`](../win.h) 和 [`win.c`](../win.c)，但现有构建集成只覆盖 C：

- `make win64` 构建 C `run.exe` 和 `runq.exe`；
- `build_msvc.bat` 只构建 `run.c`；
- Makefile 没有链接 `win.c` 的 C++ Windows target；
- CI 不构建 C++ Windows 程序。

源码中的 `_WIN32` include 分支不能单独证明 C++ Windows 路径已受支持。

### 4.5 测试和 CI 范围

仓库现有自动测试没有覆盖：

- `run.cpp`；
- `runq.c` 或 `runq.cpp` 的端到端输出；
- shared/unshared Wcls 和 MHA/GQA/MQA 的参数化组合；
- checkpoint 截断、损坏、尾随数据或跨端序；
- Chat 安全性与多轮正确性。

[`build.yml`](../.github/workflows/build.yml) 的 Ubuntu 和 macOS job 构建 C 并运行 pytest。Windows MSVC job 通过 batch 文件构建 `run.c`；MinGW job 构建 C `run.exe` 和 `runq.exe`。非交叉编译的 Windows job 也运行 pytest。

Workflow 的 path filter 包含 `*.c`、`*.h`、`*.py`、Makefile 和 workflow 文件，但不包含 `*.cpp`。即使 workflow 因其他文件变化而运行，现有 job 也不调用 `runcpp` 或 `runqcpp`。

> [!NOTE]
> “C++ 文件能在一台开发机上构建”与“仓库 CI 持续验证 C++ 文件”是不同结论。当前 CI 只持续覆盖 C 路径。

### 4.6 可复现性和性能

Greedy sampling 不使用随机数。随机路径若要比较 C 与 C++，应显式传入相同的正 seed，并固定 checkpoint、prompt、编译器和浮点选项。不传 `-s` 时，两种实现的自动 seed 来源不同。`-ffast-math` 也可能改变 logits 和采样结果。

`achieved tok/s` 从第一轮之后开始计时，意图排除部分初始化开销。该值仍混合 prompt processing 和 decode，也不分别报告首 token 延迟、prefill throughput 与 decode throughput。

比较性能时，应固定模型、prompt 长度、生成长度、线程数、编译选项、CPU 状态和采样方式。Q8 checkpoint 更小，但当前实现没有硬件专用 SIMD packing，因此不保证在所有 CPU 上快于优化后的 FP32 实现。

### 4.7 相关文件

- FP32 C Runtime：[`run.c`](../run.c)
- FP32 C++ Runtime：[`run.cpp`](../run.cpp)
- Q8 C Runtime：[`runq.c`](../runq.c)
- Q8 C++ Runtime：[`runq.cpp`](../runq.cpp)
- 模型导出：[`export.py`](../export.py)
- Python/端到端测试：[`test_all.py`](../test_all.py)
- C tokenizer 测试：[`test.c`](../test.c)
- 构建入口：[`Makefile`](../Makefile)
- CI 配置：[`build.yml`](../.github/workflows/build.yml)
