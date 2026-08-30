# llama2.cpp

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

本仓库基于 Karpathy 的 [llama2.c](https://github.com/karpathy/llama2.c)，保留原始 C/Python 训练与推理代码，并增加两份可独立运行的 C++20 推理实现：

| 程序 | Checkpoint | 权重与线性层 | 定位 |
| --- | --- | --- | --- |
| [run.c](run.c) | legacy v0 | FP32 | 上游 C reference |
| [run.cpp](run.cpp) | legacy v0 | FP32 | C++20 reference |
| [runq.c](runq.c) | v2 | Q8_0 | 上游 C 量化 reference |
| [runq.cpp](runq.cpp) | v2 | Q8_0 | C++20 量化 reference |

C++ 版本使用 RAII、`std::vector` 和 `std::span` 管理资源与视图，但仍保持单文件、单进程、逐 token CPU 推理，不引入服务框架、GPU 或分布式执行。

完整的模型结构、模块职责、forward 数据流、KV Cache、GQA、RoPE、SwiGLU、采样、checkpoint 与 Q8_0 原理见 [推理 Runtime 知识文档](doc/inference_runtime.md)。

## 快速运行 FP32 C++ 版本

下载上游训练的 15M 参数 TinyStories checkpoint。15M 指参数量；FP32 文件约 60 MB。

```bash
curl -L \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin \
  -o stories15M.bin

make runcpp

./run_cpp stories15M.bin \
  -i "Once upon a time" \
  -t 0 \
  -n 128
```

随机采样推荐从 `-t 1.0 -p 0.9` 开始；使用 `-s` 固定随机种子：

```bash
./run_cpp stories15M.bin \
  -i "Once upon a time" \
  -t 1.0 \
  -p 0.9 \
  -s 42 \
  -n 128
```

## 构建目标

| 命令 | 输出 | 说明 |
| --- | --- | --- |
| `make run` | `run`、`runq` | 构建上游 C FP32/Q8 程序 |
| `make runcpp` | `run_cpp` | 构建 C++20 FP32 程序 |
| `make runqcpp` | `runq_cpp` | 构建 C++20 Q8_0 程序 |
| `make runcppdebug` | `run_cpp` | C++ debug build |
| `make runcppfast` | `run_cpp` | C++ fast-math build |
| `make clean` | — | 删除本地构建产物 |

当前 C++ 版本已在 macOS 验证。Linux 使用相同 POSIX 路径，但尚未在本仓库中实机验证；C++ Windows 路径尚未完成。

## Q8_0 版本

`runq.c` 与 `runq.cpp` 不能读取上面的 v0 FP32 `stories15M.bin`，它们要求由 [export.py](export.py) 生成的 v2 Q8_0 checkpoint。

从本项目的 PyTorch checkpoint 导出：

```bash
python export.py model_q80.bin \
  --version 2 \
  --checkpoint out/ckpt.pt

make runqcpp

./runq_cpp model_q80.bin \
  -i "Once upon a time" \
  -t 0 \
  -n 128
```

Q8_0 将大矩阵权重和矩阵乘输入按组量化为 int8，但 RMSNorm、RoPE、Attention、KV Cache、残差、SwiGLU 和 logits 仍使用 FP32。详细格式与计算见 [推理 Runtime 知识文档](doc/inference_runtime.md)。

## CLI 参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-z PATH` | tokenizer binary | `tokenizer.bin` |
| `-i TEXT` | 初始 prompt | 空字符串 |
| `-n N` | 最大总 position 数 | 256 |
| `-t T` | temperature；零表示 greedy | 1.0 |
| `-p P` | top-p 阈值 | 0.9 |
| `-s SEED` | 随机种子 | 系统随机值 |
| `-m MODE` | `generate` 或 `chat` | `generate` |
| `-y TEXT` | Chat 模式的 system prompt | 空字符串 |

Chat 模式只适合相应的 instruct/chat checkpoint：

```bash
./run_cpp llama2_7b_chat.bin -m chat
```

## 模型、训练与导出

上游 TinyStories 模型：

| 模型 | `dim` | `n_layers` | `n_heads` | `n_kv_heads` | 上下文 | 参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stories260K | 64 | 5 | 8 | 4 | 512 | 260K |
| stories15M | 288 | 6 | 6 | 6 | 256 | 15M |
| stories42M | 512 | 8 | 8 | 8 | 1024 | 42M |
| stories110M | 768 | 12 | 12 | 12 | 1024 | 110M |

下载地址见 [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)。260K 模型的训练配置与预期输出见 [stories260K](doc/stories260K.md)。

训练最小流程：

```bash
python tinystories.py download
python tinystories.py pretokenize
python train.py
```

导出 legacy v0 FP32：

```bash
python export.py model.bin --checkpoint out/ckpt.pt
```

Tokenizer 训练细节见 [训练 Llama tokenizer](doc/train_llama_tokenizer.md)。Meta Llama 和兼容 Hugging Face 模型可通过 `export.py --meta-llama` 或 `export.py --hf` 导入；运行 `python export.py --help` 查看完整参数。

## 测试

运行上游测试：

```bash
pip install -r requirements.txt
make run
pytest
```

只运行 C tokenizer 测试：

```bash
make testcc
```

C++ Runtime 已针对真实 TinyStories checkpoint、MHA/GQA、shared/unshared Wcls、greedy、multinomial、AddressSanitizer 和 UndefinedBehaviorSanitizer 做过对照验证。测试不会把模型文件提交到 Git。

## 性能边界

当前 C++ kernels 是普通 CPU 标量循环，没有 OpenMP、显式 SIMD、BLAS、GPU 或 FlashAttention。该项目的目标是提供可读的 correctness reference，而不是替代 [llama.cpp](https://github.com/ggerganov/llama.cpp)、vLLM 或 SGLang。

## 上游文档与许可证

原始项目、训练脚本、模型和设计来自 [karpathy/llama2.c](https://github.com/karpathy/llama2.c)。本仓库沿用 [MIT License](LICENSE)。
