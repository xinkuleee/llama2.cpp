# Llama 2 CPU 推理运行时

本仓库用四个单文件程序展示同一条 Llama 2 推理链路：文本先被编码为 token，token 依次经过多层 Transformer，最终得到下一个 token 的分数并继续生成。两份 C 程序保留上游实现；两份 C++20 程序在不改变模型计算结构的前提下，补充资源所有权、长度信息和错误检查。

理解这套实现需要先区分两个结构层级：模型由多个 Transformer 层沿深度方向串行组成；每一层的 Attention 内部又包含多个并行 head。层决定前后依赖，head 决定单层 Attention 如何拆分特征。Checkpoint 的权重排列、FP32 与 Q8_0 的存储差异，以及 C 与 C++ 的接口设计，都建立在这两个层级之上。

文档以当前仓库源码为准：

- [run.c](../run.c)：C、FP32、legacy v0 checkpoint。
- [run.cpp](../run.cpp)：C++20、FP32、legacy v0 checkpoint。
- [runq.c](../runq.c)：C、Q8_0、v2 checkpoint。
- [runq.cpp](../runq.cpp)：C++20、Q8_0、v2 checkpoint。
- [export.py](../export.py)：从 PyTorch、Meta Llama 或 Hugging Face 输入导出 checkpoint。
- [Makefile](../Makefile)：本地构建入口。
- [test_all.py](../test_all.py)、[test.c](../test.c)：当前自动化测试入口。

---

## 1. 模型与运行时概览

### 1.1 运行时完成什么工作

Runtime 接收一个 token ID 和该 token 在序列中的位置，执行一次 Transformer 前向计算，并返回下一个 token 的 logits。

Logits 是词表中每个候选 token 的未归一化分数。Sampler 可以直接选择最大 logit，也可以先将 logits 转换为概率，再按概率采样。

一次完整运行由以下对象协作完成：

| 对象 | 职责 |
| --- | --- |
| `Config` | 保存模型维度、层数、head 数、词表大小和最大序列长度 |
| `MappedFile` | 在 C++ 版本中映射 checkpoint，并管理映射生命周期 |
| `WeightCursor` | 按 checkpoint 布局依次取得权重区域，并检查截断 |
| `TransformerWeights` | 保存所有权重的非拥有视图 |
| `RunState` | 拥有 activation、Attention 临时缓冲区、logits 和 KV Cache |
| `Transformer` | 组合配置、映射、权重视图和运行状态，并提供 `forward()` |
| `Tokenizer` | 在 UTF-8 文本和 token ID 之间转换 |
| `Sampler` | 从 logits 选择下一个 token |
| `generate()` / `chat()` | 驱动 prompt processing 和自回归生成循环 |

### 1.2 四个可执行程序

| 程序 | 语言 | Checkpoint | 大矩阵计算 | 主要用途 |
| --- | --- | --- | --- | --- |
| `run.c` | C | v0 | FP32 矩阵乘向量 | 上游最小 FP32 reference |
| `run.cpp` | C++20 | v0 | FP32 矩阵乘向量 | 带 RAII、长度和输入校验的 FP32 reference |
| `runq.c` | C | v2 | Q8 权重 × Q8 activation | 上游量化 reference |
| `runq.cpp` | C++20 | v2 | Q8 权重 × Q8 activation | 带明确所有权和边界检查的量化 reference |

这四个程序实现相同的 Llama 2 Transformer 推理链路。区别集中在语言、资源管理、checkpoint 格式和线性层数值表示。

它们不是四种模型架构，也不是四个互相调用的服务。每个程序都能作为独立命令行进程运行。

### 1.3 一次运行的边界

这里的“请求”是一次本地生成或对话调用，不是 HTTP 请求。

每个进程只维护一份 `Transformer` 和一份 `RunState`。同一时刻只有一个序列推进，没有 batch、请求队列、调度器或并发隔离。

调用关系如下：

```text
CLI 参数
  -> 映射并解析 checkpoint
  -> 加载 tokenizer
  -> 创建 sampler
  -> generate() 或 chat()
       -> Tokenizer::encode()
       -> Transformer::forward(token, position)
       -> Sampler::sample(logits)
       -> Tokenizer::decode()
       -> safe_print()
```

### 1.4 模型深度与层内多头

一个 Transformer 层，也常称一个 Transformer block，由两个子层按顺序组成：

1. 多头 Attention 子层。
2. MLP 子层。

模型按层执行。第 0 层先完成自己的多头 Attention 和 MLP，第 1 层再执行自己的多头 Attention 和 MLP，依次推进到最后一层。

`n_layers` 和 `n_heads` 描述两个不同维度：

- `n_layers` 是 Transformer block 的纵向堆叠数量。
- `n_heads` 是每个 block 内部并行计算的 Query head 数量。

如果 `n_layers = 6`、`n_heads = 8`，模型包含 6 个顺序执行的 block；每个 block 的 Attention 内部有 8 个 Query head。它不是先执行 48 个 Attention，再统一执行 6 个 MLP。

Checkpoint 按权重类型集中存储所有层的参数，是磁盘布局。`forward()` 按 layer 循环执行 Attention 和 MLP，是计算顺序。两者不要求相同。

---

## 2. 工作原理

一次生成由模型加载和序列推进两部分组成。模型加载阶段将 checkpoint 映射到内存，解析配置和权重，并分配可写的运行状态。序列推进阶段则重复执行“编码输入、计算 logits、选择 token、解码输出”。

模型内部按 Transformer 层向前推进。每层先完成自己的多头 Attention，再完成自己的 MLP；该层输出随后进入下一层。Attention 内部的多个 Query head 并行查看历史 K/V，但它们仍属于同一个 Transformer 层。

Prompt 仍未处理完时，程序把已知的下一个 prompt token 送入模型，以建立每层的 KV Cache。Prompt 处理完后，Sampler 才从 logits 选择未知的下一个 token。当前实现没有独立的矩阵化 prefill：prompt processing 和 autoregressive decode 都逐 token 调用同一个 `forward(token, position)`。

### 2.1 从文本到 token

`Tokenizer` 将 UTF-8 文本编码为整数 token ID。模型不直接处理字符串；embedding 表通过 token ID 选出一个 `dim` 维训练向量，作为第一个 Transformer 层的输入。

### 2.2 从 embedding 到 logits

每个 token 的主计算路径如下：

```text
token ID
  -> embedding
  -> Transformer 层 0：多头 Attention -> MLP
  -> Transformer 层 1：多头 Attention -> MLP
  -> ...
  -> Final RMSNorm
  -> Wcls
  -> logits
```

`Wcls` 是最终词表投影，不是 Attention 内部的 `Wo`。`Wo` 在每一层中混合多个 head 的输出；`Wcls` 在所有层结束后为整个词表生成 logits。

### 2.3 从 logits 到下一个 token

Temperature 为 0 时，Sampler 直接选择最大 logit。Temperature 大于 0 时，Sampler 将 logits 缩放并转换为概率，再走完整分布采样或 top-p 采样。选中的 token 会成为下一轮 `forward()` 的输入。

### 2.4 KV Cache 如何连接多轮 forward

每层为当前 token 计算一组 Key 和 Value，并按层号与 position 写入 KV Cache。后续 token 的 Query 会读取从位置 0 到当前位置的历史 K/V。历史 Query 不会被未来位置复用，因此不缓存。

KV Cache 属于一次序列运行产生的状态，不属于 checkpoint。当前每个 `Transformer` 只有一份 KV Cache，因此不能直接并发承载多个独立请求。

## 3. 运行示例

本节提供一个最短可运行闭环：构建 `run.cpp`，加载 FP32 checkpoint，并生成可见文本。

### 3.1 预期结果

完成后，你会得到 `run_cpp`，并看到以 `Once upon a time` 开头的生成结果。

### 3.2 前置条件

- macOS 或具有 POSIX `mmap` 接口的环境。
- 支持 C++20 的 `c++` 编译器。
- `make`。
- 与 checkpoint 词表匹配的 `tokenizer.bin`。

仓库包含默认 `tokenizer.bin`。`stories15M.bin` 不在 Git 中；下载后保存在仓库根目录。

### 3.3 获取 checkpoint

运行以下命令下载 15M 参数的 TinyStories FP32 checkpoint：

```bash
curl -L \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin \
  -o stories15M.bin
```

`stories15M.bin` 是 `<CHECKPOINT_PATH>` 的具体值。15M 表示约 1500 万个参数；FP32 文件约 60 MB。

验证文件存在：

```bash
test -f stories15M.bin && echo "checkpoint ready"
```

预期输出：

```text
checkpoint ready
```

失败条件：下载被网络、代理或 Hugging Face 访问策略阻止时，命令不会得到完整 checkpoint。不要把 HTML 错误页当作模型文件。

### 3.4 构建 FP32 C++ Runtime

```bash
make runcpp
```

该命令使用 `CXX` 和 `CXXFLAGS`，生成 `run_cpp`。

验证可执行文件：

```bash
test -x run_cpp && echo "runtime ready"
```

预期输出：

```text
runtime ready
```

失败条件：编译器不支持 C++20、系统缺少 POSIX 映射接口，或 Windows 兼容路径不完整时，构建会失败。

### 3.5 运行确定性生成

```bash
./run_cpp stories15M.bin \
  -i "Once upon a time" \
  -t 0 \
  -n 32
```

参数含义：

- `stories15M.bin`：checkpoint 路径。
- `-i`：输入 prompt。
- `-t 0`：禁用随机采样，始终选择最大 logit。
- `-n 32`：最多处理 32 个 position；prompt token 也计入该上限。

预期结果：标准输出以如下文本开头，标准错误输出包含吞吐量：

```text
Once upon a time, there was a little girl ...
```

```text
achieved tok/s: <VALUE>
```

`<VALUE>` 是本机测得的每秒 token 数，不是固定值。

### 3.6 确认运行结果

如果生成文本出现，并且进程以状态码 0 退出，FP32 C++ 加载、tokenizer、forward、采样和解码链路已经连通。

### 3.7 下一步

- 了解每个计算阶段：继续阅读[一轮生成的完整计算过程](#4-一轮生成的完整计算过程)。
- 切换 Q8_0：参阅[导出并运行 v2 Q8_0](#58-导出并运行-v2-q8_0)。
- 查询参数和格式：参阅[组件与接口](#6-组件与接口)。
- 评估不支持的功能：参阅[实现边界与已知限制](#9-实现边界与已知限制)。

---

## 4. 一轮生成的完整计算过程

本节通过一次生成过程说明文本、token、模型状态和输出之间的关系。

### 4.1 将文本编码为 token

`Tokenizer::encode()` 把输入字符串转换为整数 token ID 序列。

Tokenizer 先添加可选 BOS token。BOS 表示序列开始，本实现固定使用 ID 1。对非空输入，Tokenizer 还添加 SentencePiece 风格的空格前缀。

输入文本按 UTF-8 code point 扫描。词表中存在完整 code point 时直接使用其 token；不存在时，退化为 byte fallback token。之后，Tokenizer 反复合并得分最高的相邻 token 对，直到没有可合并项。

生成模式调用：

```cpp
tokenizer.encode(prompt, true, false)
```

这表示添加 BOS，不添加 EOS。

### 4.2 处理 prompt

`generate()` 从 prompt 的第一个 token 开始，逐 position 调用：

```cpp
model.forward(token, position)
```

只要 prompt 尚未消费完，程序就忽略当前 logits 的采样结果，强制把 prompt 中的下一个 token 作为下一轮输入。

这一阶段通常称为 prompt processing 或 prefill。当前 runtime 没有矩阵化 prefill kernel；它仍然逐 token 调用与 decode 相同的 `forward()`。

每处理一个 prompt token，`forward()` 都会把每层新产生的 Key 和 Value 写入 KV Cache。KV Cache 是保存历史 K/V 向量的运行状态，使后续 token 不必重复计算历史位置的 K/V。

### 4.3 执行一次 forward

当前 token 先查 embedding 表，得到长度为 `dim` 的向量 `x`。

Embedding 是 token ID 的训练所得向量表示。它不是 token 字符串，也不是概率。

随后，模型依次执行每个 Transformer block：

```text
Attention RMSNorm
  -> Q/K/V 线性投影
  -> RoPE
  -> 写入并读取 KV Cache
  -> 多头 causal Attention
  -> Wo 输出投影
  -> 第一次残差相加
  -> FFN RMSNorm
  -> W1/W3 + SwiGLU + W2
  -> 第二次残差相加
```

一个 block 完成后，它的输出 `x` 才成为下一个 block 的输入。

### 4.4 计算 Q、K 和 V

Attention 前先对 `x` 执行 RMSNorm，得到归一化向量。RMSNorm 按向量均方根缩放特征，但不减去均值。

三个训练所得权重矩阵对同一个归一化输入做不同线性投影：

$$
Q = W_q x, \qquad K = W_k x, \qquad V = W_v x
$$

`Wq` 产生当前 token 的所有 Query head。`Wk` 和 `Wv` 产生当前 token 的所有 KV head。当前 K 和 V 直接写到本层、当前位置对应的 KV Cache 区域。

Value 不是从 Attention score 计算出来的。它由 `Wv` 对归一化输入做矩阵乘向量得到。Attention score 只决定历史 Value 的组合权重。

### 4.5 注入位置信息

RoPE 是 Rotary Position Embedding。它在每个 Q/K head 内，将相邻两个坐标视为二维向量，并按 position 相关角度旋转。

对坐标对 `(a, b)`：

$$
a' = a\cos\theta - b\sin\theta
$$

$$
b' = a\sin\theta + b\cos\theta
$$

本实现旋转 Q 和 K，不旋转 V。写入 KV Cache 的 K 已经完成 RoPE。

### 4.6 执行多头 causal Attention

每个 Query head 取得当前 position 的 Query 向量，并与位置 0 到当前位置的兼容 Key 逐一做缩放点积：

$$
\operatorname{score}_t = \frac{q \cdot k_t}{\sqrt{\operatorname{head\_size}}}
$$

程序只遍历 `0..position`，因此天然不会读取未来 token，不需要显式分配 causal mask 矩阵。

稳定 Softmax 将一行 score 转为总和为 1 的 Attention 权重：

$$
p_t = \frac{e^{\operatorname{score}_t-m}}
{\sum_j e^{\operatorname{score}_j-m}},
\qquad
m = \max_j \operatorname{score}_j
$$

随后，用这些权重对历史 Value 求加权和：

$$
o = \sum_{t=0}^{\operatorname{position}} p_t v_t
$$

每个 head 得到一个 `head_size` 长度的输出。所有 head 的输出写入 `xb` 中连续且互不重叠的片段；因此 `xb` 在内存中已经是 head 拼接结果。

`Wo` 再把拼接结果投影回 `dim`，并与残差流 `x` 相加。

### 4.7 执行 MLP

这里的 MLP 与 FFN 指同一个 Transformer 子层。FFN 全称 Feed-Forward Network。

MLP 先对 Attention 残差结果做第二次 RMSNorm，然后执行：

$$
g = W_1 x, \qquad u = W_3 x
$$

$$
\operatorname{MLP}(x) = W_2\left(\operatorname{SiLU}(g) \odot u\right)
$$

`W1` 和 `W3` 都把 `dim` 投影到 `hidden_dim`。`W1` 的输出经过 SiLU，形成门控分支；`W3` 的输出形成被门控分支。两者逐元素相乘后，`W2` 将结果投影回 `dim`。

这个组合称为 SwiGLU。`W2` 的输出与残差流再次相加。

### 4.8 产生 logits

最后一个 Transformer block 完成后，模型执行 final RMSNorm，并使用 `Wcls` 投影到词表维度：

$$
\operatorname{logits} = W_{\mathrm{cls}}\operatorname{RMSNorm}(x)
$$

`Wcls` 是 vocabulary classifier，也称 LM head 或词表输出投影。它不是 Softmax，也不是 Attention 的 `Wo`。

如果 checkpoint 使用 weight tying，`Wcls` 与 token embedding 表指向同一份权重；否则 checkpoint 额外保存独立 `Wcls`。

### 4.9 从 logits 选择 token

当 `temperature = 0` 时，Sampler 直接返回最大 logit 的索引。这是 greedy decoding，不需要词表 Softmax。

当 `temperature > 0` 时，Sampler 先用 temperature 缩放 logits，再执行稳定 Softmax，得到词表概率。

如果 `top_p` 位于 0 和 1 之间，Sampler 只保留累计概率超过 `top_p` 的最小高概率候选集合，再从该集合采样。否则，它从完整概率分布采样。

### 4.10 进入 autoregressive decode

Prompt 消费完后，Sampler 选择的 token 成为下一轮 `forward()` 的输入：

```text
上一个 token
  -> forward()
  -> logits
  -> sample()
  -> 下一个 token
  -> decode()
  -> 输出文本片段
```

这一阶段称为 autoregressive decode。每轮只生成一个 token，并复用之前积累的 KV Cache。

---

## 5. 常用操作

### 5.1 选择 Runtime 和 checkpoint

| 已有文件 | 使用程序 | 不要使用 |
| --- | --- | --- |
| legacy v0 FP32 | `run` 或 `run_cpp` | `runq`、`runq_cpp` |
| v1 FP32 | 当前没有对应本地 runtime | 四个现有 runtime |
| v2 Q8_0 | `runq` 或 `runq_cpp` | `run`、`run_cpp` |

文件扩展名都是 `.bin`，不能仅通过扩展名判断格式。

`run.cpp` 会把文件开头直接解释为 v0 `Config`。`runq.cpp` 则要求 magic `0x616b3432` 和 version 2。

### 5.2 构建四个 Runtime

构建两个 C 程序：

```bash
make run
```

输出 `run` 和 `runq`。

构建两个 C++ 程序：

```bash
make runcpp
make runqcpp
```

输出 `run_cpp` 和 `runq_cpp`。

如需指定编译器：

```bash
make runcpp CXX=clang++
```

### 5.3 运行确定性生成

```bash
./run_cpp <V0_CHECKPOINT_PATH> \
  -z <TOKENIZER_PATH> \
  -i "<PROMPT>" \
  -t 0 \
  -n 128
```

- `<V0_CHECKPOINT_PATH>`：legacy v0 FP32 checkpoint。
- `<TOKENIZER_PATH>`：与模型 token ID 分配一致的 tokenizer binary。
- `<PROMPT>`：输入文本。

### 5.4 运行随机生成

```bash
./run_cpp <V0_CHECKPOINT_PATH> \
  -z <TOKENIZER_PATH> \
  -i "<PROMPT>" \
  -t 1.0 \
  -p 0.9 \
  -s 42 \
  -n 128
```

`-s 42` 固定 xorshift64* 的初始状态。C++ 版本将非正 seed 转为 0，再使用 `std::random_device` 生成非零状态。C 版本只在未指定 seed 或解析结果为 0 时使用当前时间；显式负数会在赋给无符号状态时发生转换，不会进入自动 seed 路径。

### 5.5 运行 Chat 模式

Chat 模式要求与 Llama 2 `[INST]` 模板兼容的 instruct/chat checkpoint。

```bash
./run_cpp <CHAT_V0_CHECKPOINT_PATH> \
  -z <TOKENIZER_PATH> \
  -m chat \
  -y "<SYSTEM_PROMPT>" \
  -i "<FIRST_USER_PROMPT>"
```

如果省略 `-y`，程序首次运行时从标准输入读取 system prompt。如果省略 `-i`，程序读取第一条 user prompt。

Base model 不会因为设置 `-m chat` 自动获得对话能力；该选项只改变 prompt 模板和控制流。

### 5.6 导出 v0 FP32

```bash
python export.py <OUTPUT_V0_BIN> \
  --version 0 \
  --checkpoint <INPUT_PT_CHECKPOINT>
```

`<INPUT_PT_CHECKPOINT>` 是包含 `model_args` 和 `model` 的本项目 PyTorch checkpoint。v0 可交给 `run` 和 `run_cpp`。

### 5.7 导出 v1 FP32

```bash
python export.py <OUTPUT_V1_BIN> \
  --version 1 \
  --checkpoint <INPUT_PT_CHECKPOINT>
```

v1 增加 256-byte header，但当前四个 runtime 都不加载 v1。

### 5.8 导出并运行 v2 Q8_0

```bash
python export.py <OUTPUT_V2_BIN> \
  --version 2 \
  --checkpoint <INPUT_PT_CHECKPOINT>

make runqcpp

./runq_cpp <OUTPUT_V2_BIN> \
  -z <TOKENIZER_PATH> \
  -i "<PROMPT>" \
  -t 0 \
  -n 128
```

`version2_export()` 默认从 group size 64 开始。如果 `dim` 不能整除 64，它会反复减半，直到 `dim` 可整除。

失败条件：`dim` 或 `hidden_dim` 不能被 header 中的 group size 整除时，`runq.cpp` 拒绝加载。

### 5.9 从 Meta Llama 目录导出

```bash
python export.py <OUTPUT_BIN> \
  --version <0_OR_1_OR_2> \
  --meta-llama <META_MODEL_DIRECTORY>
```

`<META_MODEL_DIRECTORY>` 必须包含 `params.json` 和一个或多个 `consolidated.*.pth`。导入器会合并 shard，并构造本项目 `Transformer`。

### 5.10 从 Hugging Face 目录导出

```bash
python export.py <OUTPUT_BIN> \
  --version <0_OR_1_OR_2> \
  --hf <HF_MODEL_PATH>
```

此路径依赖 `transformers`，但 `requirements.txt` 当前没有声明该包。当前导入器也不保留 `num_key_value_heads`；不要直接用于 GQA/MQA 模型。

### 5.11 使用自定义 tokenizer

```bash
./run_cpp <CHECKPOINT_PATH> \
  -z <TOKENIZER_BIN> \
  -i "<PROMPT>"
```

Tokenizer 文件没有内嵌 `vocab_size`。Runtime 使用 checkpoint 的 `Config.vocab_size` 决定读取多少个词条。Checkpoint 和 tokenizer 必须匹配；词表大小相同并不足以保证 token ID 含义相同。

### 5.12 运行当前自动化测试

```bash
python -m pip install -r requirements.txt
make run
pytest
```

`pytest` 首次运行会下载 `stories260K.bin`、`stories260K.pt`、`tok512.bin` 和 `tok512.model` 到 `test/`。

只运行 C tokenizer 单元测试：

```bash
make testcc
```

预期输出：

```text
ALL OK
```

当前 pytest 和 `test.c` 不编译、不执行 `run.cpp` 或 `runq.cpp`。

### 5.13 清理构建产物

```bash
make clean
```

该目标删除 `run`、`runq`、`run_cpp`、`runq_cpp`、`testc` 以及两个 C++ dSYM 目录。它不删除 checkpoint、tokenizer、pytest 下载目录或文档。

### 5.14 排查 checkpoint 加载失败

1. `failed to open`：确认路径存在且进程有读取权限。
2. `invalid checkpoint magic`：Q8 runtime 收到的不是 v2 文件。
3. `requires a version 2 checkpoint`：文件有 header，但版本不是 2。
4. `checkpoint truncated`：文件短于 Config 所声明的权重布局。
5. `model dimensions are incompatible with group size`：v2 group size 不兼容模型维度。
6. 生成乱码：检查 tokenizer 是否与 checkpoint 匹配。

---

## 6. 组件与接口

### 6.1 CLI

| 参数 | 类型 | 默认值 | 行为 |
| --- | --- | --- | --- |
| 第一个位置参数 | 路径 | 必填 | checkpoint 路径 |
| `-z` | 路径 | `tokenizer.bin` | tokenizer binary |
| `-i` | 字符串 | 空 | generate prompt 或 chat 第一条 user prompt |
| `-n` | 整数 | 256 | 最大总 position 数；非正值表示 `seq_len` |
| `-t` | 浮点数 | 1.0 | temperature；0 使用 greedy |
| `-p` | 浮点数 | 0.9 | top-p；越界时恢复为 0.9 |
| `-s` | 整数 | 自动生成 | 随机种子；正数用于复现 |
| `-m` | 枚举 | `generate` | `generate` 或 `chat` |
| `-y` | 字符串 | 空 | chat system prompt |

每个 flag 必须紧跟一个值。C++ parser 拒绝未知选项、缺失值和未完整解析的数字。

`steps` 是 position 总预算，不是额外生成 token 数。Prompt processing 和 decode 共用该预算。

### 6.2 Makefile 目标

| 目标 | 输出 | 关键选项 |
| --- | --- | --- |
| `run` | `run`、`runq` | C、`-O3` |
| `runcpp` | `run_cpp` | C++20、warnings、`-O3` |
| `runqcpp` | `runq_cpp` | C++20、warnings、`-O3` |
| `rundebug` | `run`、`runq` | `-g` |
| `runcppdebug` | `run_cpp` | `-O0 -g` |
| `runfast` | `run`、`runq` | `-Ofast` |
| `runcppfast` | `run_cpp` | `-O3 -ffast-math` |
| `runomp` | `run`、`runq` | OpenMP、`-march=native` |
| `win64` | `run.exe`、`runq.exe` | MinGW 交叉编译 |
| `rungnu` | `run`、`runq` | GNU C11 |
| `runompgnu` | `run`、`runq` | GNU C11、OpenMP |
| `test` | — | 全部 pytest |
| `testc` | — | 仅选择 `runc` 测试名 |
| `testcc` | `testc` | C tokenizer 测试 |
| `clean` | — | 删除已知构建产物 |

`make` 的默认首个实际目标是 `run`，因此 CI 中裸 `make` 只构建 C 程序。

### 6.3 Config

`Config` 由 7 个连续 `int32_t` 字段组成。C++ 使用 `static_assert` 要求总大小为 28 bytes。

| 字段 | 含义 | 约束 |
| --- | --- | --- |
| `dim` | residual stream 和 embedding 宽度 | 正数，能被 `n_heads` 整除 |
| `hidden_dim` | MLP 中间宽度 | 正数；Q8 中需兼容 group size |
| `n_layers` | Transformer block 数量 | 正数 |
| `n_heads` | 每层 Query head 数量 | 正数 |
| `n_kv_heads` | 每层 Key/Value head 数量 | 正数，整除 `n_heads` |
| `vocab_size` | 词表大小 | v0 磁盘值的符号还编码 Wcls 共享状态 |
| `seq_len` | 最大 position 数 | 正数 |

派生量：

$$
\operatorname{head\_size} = \frac{\operatorname{dim}}{\operatorname{n\_heads}}
$$

$$
\operatorname{kv\_dim} =
\operatorname{dim}\frac{\operatorname{n\_kv\_heads}}
{\operatorname{n\_heads}}
$$

RoPE 要求 `head_size` 为偶数，因为它每次旋转两个相邻坐标。

### 6.4 MHA、GQA 和 MQA

| 类型 | 条件 | 共享关系 |
| --- | --- | --- |
| MHA | `n_kv_heads == n_heads` | 每个 Query head 有独立 KV head |
| GQA | `1 < n_kv_heads < n_heads` | 一组 Query head 共享一个 KV head |
| MQA | `n_kv_heads == 1` | 所有 Query head 共享一个 KV head |

Runtime 使用 `kv_mul = n_heads / n_kv_heads` 和 `kv_head = query_head / kv_mul` 完成映射。

例如 `n_heads = 8`、`n_kv_heads = 4` 时，Query head 0 和 1 使用 KV head 0，Query head 2 和 3 使用 KV head 1，以此类推。

这些数量由模型架构和训练权重决定。推理时不能只修改 Config；`Wk`、`Wv` 的形状和训练语义也必须匹配。

### 6.5 C++ 所有权与生命周期

| 类型 | 拥有数据 | 引用数据 | 生命周期要求 |
| --- | --- | --- | --- |
| `MappedFile` | 映射资源 | checkpoint bytes | 必须活得比所有权重 view 久 |
| `WeightCursor` | 无 | mapped payload | 只在解析阶段使用 |
| `TransformerWeights` | 无 | mapped weights | 依赖 `MappedFile` |
| `QuantizedTensor` | 无 | int8 values 和 FP32 scales | 依赖映射或 owning buffer |
| `QuantizedBuffer` | 两个 vector | `view()` 返回自身 view | view 不得超过 buffer 生命周期 |
| `RunState` | 所有可写 vector | — | 与单个 `Transformer` 同寿命 |
| `Transformer` | 映射、配置、状态和描述符 | — | 完整模型生命周期边界 |

`std::span` 是指针和长度组成的非拥有视图。创建 span 不复制权重，也不延长底层内存生命周期。

### 6.6 MappedFile

`MappedFile` 构造时以只读方式打开 checkpoint，用 `fstat` 取得大小，拒绝空文件，再使用 `mmap(..., PROT_READ, MAP_PRIVATE, ...)` 建立私有映射。映射成功后关闭文件描述符。

析构函数调用 `munmap`。复制和移动都被禁用，避免多个对象不清楚地管理同一映射。

### 6.7 WeightCursor

FP32 版本读取 `span<const float>`。`take(count, name)` 返回当前位置开始的权重，并推进游标。

Q8 版本读取 `span<const byte>`：

- `take_fp32()` 取得 FP32 区域并检查对齐。
- `take_q80()` 依次取得 int8 values 和每组一个 FP32 scale。
- `take_bytes()` 在每次读取前检查剩余字节数。

游标只负责磁盘布局解析，不拥有模型，也不执行推理。

### 6.8 TransformerWeights

| 权重 | 每层或全局形状 | 作用 |
| --- | --- | --- |
| `token_embedding_table` | `vocab_size × dim` | token ID 到 embedding |
| `rms_att_weight` | `n_layers × dim` | Attention 前 RMSNorm scale |
| `wq` | `n_layers × dim × dim` | Query 投影 |
| `wk` | `n_layers × kv_dim × dim` | Key 投影 |
| `wv` | `n_layers × kv_dim × dim` | Value 投影 |
| `wo` | `n_layers × dim × dim` | 多头输出投影 |
| `rms_ffn_weight` | `n_layers × dim` | MLP 前 RMSNorm scale |
| `w1` | `n_layers × hidden_dim × dim` | SwiGLU 门控投影 |
| `w2` | `n_layers × dim × hidden_dim` | MLP 下投影 |
| `w3` | `n_layers × hidden_dim × dim` | SwiGLU 上投影 |
| `rms_final_weight` | `dim` | 最终 RMSNorm scale |
| `wcls` | `vocab_size × dim` | 词表输出投影 |

Q8 版本保留三个 RMSNorm 权重区域为 FP32，其余大矩阵使用 `QuantizedTensor`。

### 6.9 RunState

| 缓冲区 | 大小 | 用途 |
| --- | ---: | --- |
| `x` | `dim` | 当前 residual stream |
| `xb` | `dim` | 归一化输入、head 拼接结果或 MLP 输出 |
| `xb2` | `dim` | Attention 输出投影 |
| `hb` | `hidden_dim` | W1 输出和 SwiGLU 结果 |
| `hb2` | `hidden_dim` | W3 输出 |
| `q` | `dim` | 当前 token 的全部 Query head |
| `attention` | `n_heads × seq_len` | score；Softmax 后复用为权重 |
| `logits` | `vocab_size` | 下一个 token 的未归一化分数 |
| `key_cache` | `n_layers × seq_len × kv_dim` | 所有层的历史 Key |
| `value_cache` | 同上 | 所有层的历史 Value |
| `xq` | Q8 `dim` | 动态量化的线性层输入 |
| `hq` | Q8 `hidden_dim` | 动态量化的 W2 输入 |

`key_at(config, layer, position)` 和 `value_at(...)` 返回当前层、当前位置长度为 `kv_dim` 的 span，将 KV Cache 布局集中在 `RunState` 内。

### 6.10 Transformer::forward

```cpp
std::span<float> forward(std::int32_t token, std::size_t position);
```

输入契约：`token` 位于 `[0, vocab_size)`，`position < seq_len`，同一序列的 position 应从 0 递增。

返回值引用 `RunState.logits`。下一次 `forward()` 会覆盖这块内存，调用者不能把它当成长期拥有的结果。

### 6.11 FP32 kernels

| Kernel | 输入 | 输出 | 实现 |
| --- | --- | --- | --- |
| `rms_norm` | input、scale | output | FP32 标量循环 |
| `softmax_inplace` | score 或 logits | 同一缓冲区内的概率 | max-shift 稳定三遍 Softmax |
| `matmul` | row-major matrix、vector | vector | 逐行点积 |
| `apply_rope` | Q、K、position | 原地修改 Q/K | 每个 head 内二维旋转 |

`softmax_inplace` 不是 Online Softmax。当前 C++ kernel 没有 OpenMP、BLAS、显式 SIMD、GPU 或 FlashAttention。

### 6.12 Q8_0 数据模型和 kernel

`QuantizedTensor` 是只读、非拥有 view：

```cpp
struct QuantizedTensor {
  std::span<const std::int8_t> values;
  std::span<const float> scales;
};
```

`QuantizedBuffer` 拥有 activation 的 values 和 scales。`view()` 返回 `QuantizedTensor`，使 checkpoint 权重与动态量化 activation 共用 `matmul_q80()` 接口。

每个连续 group 共享一个 scale：

$$
s = \frac{\max_i |x_i|}{127},
\qquad
q_i = \operatorname{clamp}(\operatorname{round}(x_i/s), -127, 127)
$$

反量化为：

$$
x_i \approx q_i s
$$

`runq.cpp` 对全零 activation group 显式写入零 scale 和零 values。

Q8 矩阵乘向量先做 int8 × int8 的 `int32_t` 点积，再乘权重 scale 和 activation scale：

$$
y_r = \sum_g
\left(\sum_{k\in g}q^W_{r,k}q^x_k\right)
s^W_{r,g}s^x_g
$$

最终输出仍是 FP32。RoPE、Attention score、Softmax、Value 加权、KV Cache、残差、SwiGLU 和 logits 也保持 FP32。

`runq.c` 启动时反量化整张 embedding 表；`runq.cpp` 每次只反量化当前 token 的一行。`runq.c` 先计算临时 K/V 再复制到 cache；`runq.cpp` 让 Wk/Wv 直接写入当前 cache span。

### 6.13 Tokenizer binary

```text
uint32 max_token_length
重复 vocab_size 次：
  float32 score
  uint32 piece_length
  byte[piece_length] piece
```

文件不包含 `vocab_size`，加载器从 checkpoint 获取。特殊 token 约定为 unknown 0、BOS 1、EOS 2，byte fallback 从 3 开始。

编码路径为 dummy prefix、UTF-8 code point、byte fallback、最高分相邻 BPE merge。解码路径为 token ID 查词表、BOS 后移除一个前导空格、解析 `<0xHH>` byte token。

`safe_print()` 只过滤单字节且不可打印、非空白的控制字符，不是完整 Unicode 校验器。

### 6.14 Sampler

`Sampler::sample()` 会原地修改 logits。

| 条件 | 路径 |
| --- | --- |
| `temperature == 0` | `argmax(logits)` |
| `temperature > 0` 且 top-p 关闭 | temperature、Softmax、完整分布采样 |
| `temperature > 0` 且 `0 < top_p < 1` | temperature、Softmax、top-p 采样 |

Top-p 先用 `(1 - top_p) / (vocab_size - 1)` 排除不可能进入 nucleus 的低概率 token，再按概率降序排列候选项。

随机数生成器是 xorshift64*。零状态是吸收态，因此 C++ 构造函数会生成 seed，直到状态非零。

### 6.15 generate 和 chat

`generate()` 使用同一个循环完成 prompt processing 和 decode：prompt 尚未消费完时强制选择下一个 prompt token；之后从 logits 采样。C 和 C++ 都把 `next_token == 1` 作为 generate 结束条件。

首轮之后开始统计吞吐量，以减少初始化开销影响。

Chat 首轮模板：

```text
[INST] <<SYS>>
<SYSTEM_PROMPT>
<</SYS>>

<USER_PROMPT> [/INST]
```

后续 user turn 使用 `[INST] <USER_PROMPT> [/INST]`。EOS 结束 assistant turn。C++ 仍把 EOS 喂入 `forward()`，让它进入 KV Cache，然后开始下一条 user prompt。

## 7. Checkpoint 与权重格式

Checkpoint 保存模型配置和训练所得权重，不保存当前 prompt、KV Cache 或采样器状态。版本决定 header、权重精度和 payload 排列；文件扩展名本身不能区分版本。

### 7.1 Checkpoint v0

v0 没有 magic、version 或固定长度 header：

```text
Config: 7 × int32
token_embedding_table
rms_att_weight, wq, wk, wv, wo
rms_ffn_weight, w1, w2, w3
rms_final_weight
legacy RoPE cosine table
legacy RoPE sine table
optional Wcls
```

以上 payload 全部为 FP32。磁盘 `vocab_size > 0` 表示 Wcls 与 embedding 共享；`vocab_size < 0` 表示文件末尾有独立 Wcls。Runtime 取绝对值得到真实词表大小。

当前 forward 动态计算 RoPE，因此跳过两张 legacy RoPE 表。

### 7.2 Checkpoint v1

v1 使用 256-byte header 和 FP32 payload：

```text
offset 0: uint32 magic = 0x616b3432
offset 4: int32 version = 1
offset 8: Config
offset 36: uint8 shared_classifier
其余到 offset 255: padding
offset 256: FP32 payload
```

`export.py` 可以写 v1，但当前四个 runtime 都不加载 v1。

### 7.3 Checkpoint v2

| Offset | 类型 | 含义 |
| ---: | --- | --- |
| 0 | `uint32` | magic `0x616b3432` |
| 4 | `int32` | version 2 |
| 8 | 7 × `int32` | Config |
| 36 | `uint8` | `shared_classifier` |
| 37 | `int32` | quantization group size |
| 41..255 | bytes | padding |

Payload 按权重类型排列，顺序为：

```text
所有层 rms_att_weight: FP32
所有层 rms_ffn_weight: FP32
rms_final_weight: FP32
token_embedding_table: Q8_0
所有层 wq: Q8_0
所有层 wk: Q8_0
所有层 wv: Q8_0
所有层 wo: Q8_0
所有层 w1: Q8_0
所有层 w2: Q8_0
所有层 w3: Q8_0
optional wcls: Q8_0
```

每个 Q8_0 tensor 先保存全部 `int8[element_count]` values，再保存 `float32[element_count / group_size]` scales。

Offset 37 不保证 `int32` 对齐。`runq.cpp` 通过 `memcpy` 读取 header scalar，避免直接解引用未对齐指针。

## 8. 实现关系与设计说明

### 8.1 C 与 C++ 映射

| C | C++ | 变化 |
| --- | --- | --- |
| `malloc_run_state/free_run_state` | `RunState` 和 vector 析构 | 自动释放缓冲区 |
| `fd/data/file_size` | `MappedFile` | 映射生命周期集中管理 |
| `memory_map_weights` 裸指针推进 | `WeightCursor` | 每次切片检查长度 |
| 权重裸指针 | `span<const T>` | 增加长度，不复制数据 |
| `build_transformer` | `Transformer` 构造函数 | 构造完成即满足对象不变量 |
| `free_transformer` | RAII 析构 | 异常路径也释放资源 |
| 全局 Q8 `GS` | `Transformer::group_size_` | 成为单模型状态 |
| C Q8 descriptor | `QuantizedTensor` view | 明确非拥有范围 |
| xq/hq 裸分配 | `QuantizedBuffer` | 明确拥有可写存储 |
| `exit()` | exception + `main()` catch | 自动展开并清理 |
| 临时 K/V 后复制 | `key_at()/value_at()` 直接写 cache | 消除两次复制 |

C++ 重写没有改变模型权重的训练含义，也没有增加 GPU、batch 或服务调度。

### 8.2 测试与 CI

`test_all.py` 包含：

- `test_runc`：运行 C `./run` 200 steps，与固定 ASCII 输出逐字节比较。
- `test_python`：运行 `model.py` 的 PyTorch 模型，与同一固定输出比较。

`test.c` 通过 `#include "run.c"` 编译 C 实现，并验证五组 tokenizer 编码结果。它不测试 forward、Q8 或 C++。

GitHub Actions 当前执行：

- Ubuntu：`make`、`make runfast`、pytest。
- macOS：`make run CC=clang`、`make`、`make runfast`、pytest。
- Windows MSVC：`build_msvc.bat` 构建 C `run.c`，可运行架构执行 pytest。
- Windows MinGW：`make win64` 构建两个 C 程序，执行 pytest。

Workflow path filter 不包含 `.cpp` 和 `.md`。CI 也不调用 `make runcpp` 或 `make runqcpp`。

### 8.3 本页验证记录

本页生成时在当前 macOS 工作区完成：

- `run.c` 和 `runq.c` 的 C11 语法编译。
- `runq.cpp` 的 C++20 warning-enabled 语法编译。
- `run.cpp` 的 C++20 `-O3` 构建。
- `run_cpp stories15M.bin -i "Once upon a time" -t 0 -n 12` 冒烟运行。

实测生成前缀：

```text
Once upon a time, there was a little girl named L
```

这是一轮本地验证，不替代仓库中缺失的持久化 C++ 自动测试。

---

### 8.4 为什么按层交替执行 Attention 和 MLP

第 `layer` 个 block 有自己训练得到的 Attention、RMSNorm 和 MLP 权重。其数据依赖为：

```text
x_layer
  -> Attention_layer
  -> Attention residual
  -> MLP_layer
  -> x_(layer + 1)
```

`MLP_layer` 的输入依赖同层 Attention 的输出；下一层 Attention 又依赖本层 MLP 的输出。Checkpoint 可以先保存所有层的 `wq` 再保存所有层的 `wk`，因为磁盘顺序只决定参数位置，不决定执行顺序。

### 8.5 为什么 Attention 使用多个 head

`dim` 是每个 token 在 residual stream 中的特征宽度。多头 Attention 将 Query 维度划分为 `n_heads` 个长度为 `head_size` 的子向量。

每个 head 独立计算当前 Query 对历史 Key 的权重，再组合历史 Value。所有 head 输出拼接后仍为 `dim`，再由 `Wo` 混合并投影回 residual stream。

Head 是一层内部的并行分解；layer 是模型深度。两者不能相互替代。

### 8.6 为什么 GQA 减少 KV head

MHA 为每个 Query head 保存独立 K/V。GQA 让多个 Query head 共享 KV head，从而减少 `Wk`/`Wv` 参数、KV Cache 元素和 decode 读取带宽。

Query head 数保持不变，因此仍可保留多个 Query 子空间。MQA 是共享程度最大的情况。

### 8.7 为什么缓存 K/V，不缓存 Q

当前 Query 只用于本轮查询。历史 Key 和 Value 会被未来 token 反复使用，因此写入 KV Cache；历史 Query 没有同样的复用收益。

Checkpoint 保存训练所得参数。KV Cache 保存当前序列执行产生的状态，两者生命周期不同。

### 8.8 Wo 与 Wcls 的区别

`Wo` 是每个 block 内 Attention 的输出投影，将各 head 的拼接结果混合成 `dim` 维 residual 更新。

`Wcls` 位于全部 block 之后，把最终隐藏向量映射为 `vocab_size` 个 logits。两者形状、位置和用途都不同。

### 8.9 为什么复用 Attention buffer

实现先把缩放点积分数写入 `attention`。Softmax 原地覆盖这块缓冲区，生成权重。后续只需要权重，因此无需同时保留原始 score。

该实现保存完整的一行 score，不是 FlashAttention 的分块 Online Softmax。

### 8.10 为什么 FP32 和 Q8 共用模型结构

量化改变数值表示和矩阵乘实现，不改变层拓扑。Q8 把大矩阵和线性层输入量化，再用整数点积和 scale 恢复 FP32 输出；非线性和 KV 状态保持 FP32。

### 8.11 为什么区分 QuantizedTensor 与 QuantizedBuffer

Checkpoint 权重已存在于映射文件中，不应复制，所以用只读非拥有 `QuantizedTensor`。

Activation 每轮动态量化，需要可写且可复用的存储，所以用拥有内存的 `QuantizedBuffer`。`view()` 只创建 span，不复制 values 或 scales。

### 8.12 为什么 C++ 代码更长

C++ 版本增加的是契约，不是新的模型功能：Config 合法性、文件大小、权重边界、span 形状、CLI 数字、tokenizer 条目、sampler 参数、RAII 和异常路径检查。

C 使用裸指针、隐式长度和进程级 `exit()`，代码更短，但许多前置条件仅由调用者保证。

### 8.13 复杂度与内存

当前位置为 `T` 时，当前 token 的 Attention 访问 0 到 `T` 的历史 K/V，因此单 token Attention 工作量随上下文长度线性增加。从空序列生成长度 `N` 时，历史扫描总量呈二次增长。

FP32 KV Cache 字节数：

$$
2 \times \operatorname{n\_layers}
\times \operatorname{seq\_len}
\times \operatorname{kv\_dim}
\times \operatorname{sizeof(float)}
$$

Q8 runtime 没有量化 KV Cache，该部分内存不变。

Q8_0 每组 `G` 个值占 `G + 4` bytes；FP32 占 `4G` bytes。`G = 64` 时，大矩阵理论存储比约为：

$$
\frac{4\times64}{64+4} \approx 3.76
$$

这只描述存储，不保证相同倍数的端到端加速。

### 8.14 为什么 prefill 与 decode 共用 forward

两阶段都需要把 token 写入每层 KV Cache，并产生 logits。当前最小接口每次只处理一个 token：prefill 强制使用已知 prompt token，decode 使用 Sampler 选择未知 token。

生产引擎通常矩阵化处理 prompt。本项目没有单独的 prefill kernel。

### 8.15 C++ 错误处理边界

`Transformer` 的成员顺序保证 `MappedFile` 在权重 view 建立前创建，并在 view 不再使用后析构。

解析失败时，已构造成员自动清理。`main()` 捕获 `std::exception`，打印错误并返回失败状态。

---

## 9. 实现边界与已知限制

### 运行与部署范围

当前实现是单进程、单序列、逐 token CPU reference，不实现：

- GPU kernel 或 FlashAttention。
- BLAS 或显式 SIMD 的 C++ kernel。
- Batch、continuous batching、请求队列或调度器。
- HTTP/RPC 服务。
- Paged KV Cache 或 quantized KV Cache。
- Tensor/Pipeline/Data Parallel。
- Speculative decoding。
- 多模型管理。

后果：该 runtime 适合学习、格式验证和小模型本地推理，不能按 vLLM、SGLang、TensorRT-LLM 或 llama.cpp 的生产吞吐能力评估。

### 平台支持

C 路径包含 POSIX 与 `win.c` Windows 映射兼容层，并由现有 CI 覆盖多个系统。

C++ 文件包含 `_WIN32` 分支，但 Makefile 和 `build_msvc.bat` 没有 C++ Windows 构建目标，CI 也不编译 C++。

后果：C++ Windows 支持尚未经过验证。

### Checkpoint 版本不能自动识别

`run.cpp` 固定读取 v0，`runq.cpp` 固定读取 v2。没有统一 loader。v1 虽可导出，但无本地 consumer。

后果：把 v2 交给 FP32 runtime 时，magic 会被误当成 `Config.dim`；把 v0 交给 Q8 runtime 时会触发 magic 错误。

### Native endian 与类型假设

`export.py` 的 `struct.pack` 未指定固定端序，runtime 也按宿主机原生端序读取。v0 C 还假设 `int` 为 32 位；C++ 明确使用 `int32_t`，但 float 表示和端序仍是本机约定。

后果：checkpoint 不是已定义的跨端序可移植格式。

### 没有 checksum

Header 没有 payload 长度、checksum、hash 或 schema fingerprint。

C++ `WeightCursor` 能检测文件短于预期，但不能证明内容正确。位翻转、错误模型但相同布局或 tokenizer 不匹配可能在加载成功后产生错误结果。

### 截断和尾随字节

C++ loader 对每个权重切片检查剩余长度；C loader 没有等价的逐区域检查。

两个 C++ loader 完成映射后都没有要求 `cursor.remaining() == 0`。尾随字节会被忽略；C 版本面对截断文件时可能越界访问。

### 硬编码 RMSNorm epsilon

四个 native runtime 都使用 `1e-5`。该值不在 v0/v1/v2 Config 中。源模型使用其他 epsilon 时，导出成功不代表 native forward 数值等价。

### 硬编码 RoPE theta

四个 native runtime 都使用 `10000.0`。Checkpoint 不保存 `rope_theta` 或 rope scaling。

使用其他 theta、NTK scaling、YaRN 或其他位置编码配置的模型不会被忠实执行。v0 保存的两张 legacy RoPE 表也被跳过并动态重算。

### export.py 导入与导出限制

本项目 `.pt` 输入要求 `model_args` 和 `model`；加载使用 `strict=False`，未匹配键不一定立即失败。

Meta Llama 导入要求 `params.json` 和 `consolidated.*.pth`，并把 `max_seq_len` 硬编码为 2048。

Hugging Face 导入依赖未列入 `requirements.txt` 的 `transformers`。它把 `n_kv_heads` 设置为 `num_attention_heads`，忽略 `num_key_value_heads`，因此不适合直接转换 GQA/MQA checkpoint。

`--dtype` 只对 `version = -1` 的 Hugging Face 输出有意义。命令行末尾当前将 `args.dtype` 字符串传给 `model_export()`，不是已解析的 torch dtype；这影响 version -1，不影响 v0/v1/v2。

### v2 group size

`version2_export()` 只根据 `dim` 自动回退 group size，然后要求所有 tensor 元素数可整除。Runtime 还要求 `hidden_dim` 可整除。

导出断言错误消息引用未在该循环定义的 `i`；触发不兼容 tensor 时可能得到 `NameError`，而不是清晰断言。

### Q8 数值与性能

Q8_0 是有损量化。离群值会降低同组其他值的有效精度。`runq.cpp` 使用普通标量整数循环，没有专用 SIMD 或平台量化库。

后果：文件更小不保证端到端更快，结果也不保证与 FP32 逐位一致。

`runq.c` activation quantize 和 `export.py` 权重量化没有显式处理全零 group；`runq.cpp` 已处理该情况。

### Tokenizer 兼容范围

实现是面向本项目 Llama 2 SentencePiece BPE binary 的最小 tokenizer，不是通用 SentencePiece runtime。

它不自描述 vocab size，假设 byte fallback 从 ID 3 开始，不读取完整 normalization 配置，也不执行完整 UTF-8 输出校验。

后果：不同特殊 token、normalizer 或 byte fallback 布局可能加载成功但产生错误 token。

### Sampler 语义

Sampler 原地覆盖 logits。Top-p 需要筛选并排序候选项，没有大词表并行实现。

C++ seed 为 0 时使用 `std::random_device`，C 使用当前时间。固定相同 seed 也不保证跨精度、编译器和实现逐 token 一致。

### Prompt processing

Prefill 与 decode 都逐 token 调用 `forward()`，没有矩阵化 prompt 路径或 prompt cache。

`-n` 同时包含 prompt token 和生成 token；prompt 达到预算时，可能没有空间生成期望数量的新 token。

### Chat 模式

Chat 模板固定为 Llama 2 `[INST]`，不支持自动模板选择、工具调用、结构化消息或多角色 schema。

C 源码明确把 chat 标为 proof of concept，并使用固定大小 `system_prompt[512]`、`user_prompt[512]`、`rendered_prompt[1152]` 和 token buffer，以及 `strcpy`/`sprintf`。

后果：过长或不受信任的输入可能造成 C chat 缓冲区溢出。C++ 使用 `std::string` 消除了这些固定字符缓冲区，但仍是简化状态机。

### KV Cache 生命周期

`RunState` 只有一份 KV Cache，没有 request ID、sequence ID 或 `reset()` API。

后果：同一 `Transformer` 实例不能直接作为并发多请求执行器；复用于无关序列前必须重新建立正确状态。

### 自动测试缺口

仓库没有持久化测试覆盖 C++ FP32/Q8 真实输出、C/C++ 逐步 logits、shared/unshared Wcls、完整 MHA/GQA/MQA 矩阵、截断/尾随字节、端序、checksum 或 sanitizer CI。

修改 C++ Runtime 后，应显式运行 C++ 构建和真实 checkpoint 冒烟测试。

### CI 触发缺口

Workflow path filter 不包含 `.cpp` 和 `.md`，job 也不构建 C++。

后果：只修改 C++ 时，push 或 pull request 可能不触发 CI；手动启动现有 CI 也不能证明 C++ Runtime 正确。

### tok/s 不是基准承诺

`achieved tok/s` 从首轮之后计时，可能混合 prompt processing 与 decode，并受模型、上下文位置、编译选项、CPU、采样路径和系统负载影响。

正式比较需要固定模型、prompt、生成长度、线程、编译器和统计方法。
