<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

# SGLang - 高性能大语言模型服务框架

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)

</div>

--------------------------------------------------------------------------------

| [**博客**](https://lmsys.org/blog/)
| [**文档**](https://docs.sglang.io/)
| [**路线图**](https://roadmap.sglang.io/)
| [**加入Slack**](https://slack.sglang.io/)
| [**每周开发者会议**](https://meet.sglang.io/)
| [**幻灯片**](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides) |

## 项目简介

SGLang (SGLang Runtime) 是一个高性能的大语言模型（LLM）和视觉语言模型（VLM）服务框架。该项目旨在为大规模生产环境提供低延迟、高吞吐量的推理服务，支持从单GPU到分布式集群的各种部署场景。

SGLang的核心设计目标是：
- **高性能**: 通过先进优化技术实现业界领先的推理性能
- **易用性**: 提供直观的前端语言接口，简化LLM应用开发
- **可扩展性**: 支持多种硬件平台和模型架构
- **生产就绪**: 工业级的稳定性和监控能力

## 功能说明

### 核心功能
- **高速后端运行时**: 基于RadixAttention的前缀缓存、零开销CPU调度器、预填充-解码解耦、推测解码、连续批处理、分页注意力等
- **张量/流水线/专家/数据并行**: 支持多种并行策略，适应不同规模模型
- **结构化输出**: 支持JSON模式和有限状态机的快速结构化输出生成
- **量化支持**: 支持FP4/FP8/INT4/AWQ/GPTQ等多种量化格式
- **LoRA支持**: 高效的LoRA适配器加载和批处理
- **多模态支持**: 支持图像、视频等多模态输入处理

### 模型支持
- **生成模型**: Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral等主流模型
- **嵌入模型**: e5-mistral、gte、mcdse等
- **奖励模型**: Skywork等
- **扩散模型**: WAN、Qwen-Image等
- **MoE模型**: Mixtral、DeepSeek-MoE等

### 硬件支持
- **NVIDIA GPU**: GB200/B300/H100/A100/Spark等
- **AMD GPU**: MI355/MI300等
- **Intel平台**: Xeon CPU、XPU等
- **Google TPU**: TPU v4/v5等
- **Ascend NPU**: 华为昇腾NPU

## 技术栈

### 后端技术
- **编程语言**: Python + PyTorch
- **核心框架**: Transformers、FlashInfer
- **网络通信**: FastAPI、aiohttp、gRPC
- **并发处理**: asyncio、uvloop
- **数据处理**: numpy、tiktoken、sentencepiece

### 优化技术
- **内存管理**: 分页注意力、前缀缓存、内存池
- **计算优化**: CUDA图、Triton内核、量化
- **批处理策略**: 连续批处理、动态批处理
- **并行计算**: 张量并行、流水线并行、专家并行

### 特性支持
- **采样算法**: 贪心采样、Top-k、Top-p、温度采样
- **约束生成**: 正则表达式约束、JSON模式
- **模型服务**: 模型加载、卸载、切换
- **监控指标**: 性能监控、资源使用统计

## 项目架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端语言层 (lang/)                       │
├─────────────────────────────────────────────────────────────────┤
│     API接口 │ 函数式编程 │ 控制流 │ 多模态支持 │ 工具调用        │
├─────────────────────────────────────────────────────────────────┤
│                       服务接口层 (entrypoints/)                 │
├─────────────────────────────────────────────────────────────────┤
│           HTTP API │ gRPC API │ OpenAI兼容接口                  │
├─────────────────────────────────────────────────────────────────┤
│                       核心调度层 (managers/)                    │
├─────────────────────────────────────────────────────────────────┤
│  请求调度 │ 批处理管理 │ 内存管理 │ 负载均衡 │ 资源监控         │
├─────────────────────────────────────────────────────────────────┤
│                     模型执行层 (model_executor/)                │
├─────────────────────────────────────────────────────────────────┤
│    模型推理 │ 张量计算 │ 优化内核 │ 分布式计算 │ 内存优化        │
├─────────────────────────────────────────────────────────────────┤
│                     硬件抽象层 (hardware_backend/)              │
├─────────────────────────────────────────────────────────────────┤
│     CUDA │ ROCm │ TPU │ CPU │ 量化优化 │ 自定义算子            │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件说明

1. **前端语言 (lang)**: 提供高级编程接口，支持函数式编程、控制流等特性
2. **API服务器 (entrypoints)**: 提供HTTP/gRPC接口，兼容OpenAI API
3. **调度器 (managers)**: 负责请求调度、批处理、内存管理
4. **模型执行器 (model_executor)**: 执行模型推理计算
5. **硬件后端 (hardware_backend)**: 抽象底层硬件差异

## 目录说明

```
sglang/
├── 3rdparty/              # 第三方依赖库
├── assets/                # 静态资源文件
├── benchmark/             # 基准测试代码
├── docker/                # Docker容器配置
├── docs/                  # 项目文档
├── examples/              # 使用示例
├── notes/                 # 项目笔记和说明
├── python/                # Python源代码主目录
│   ├── sglang/
│   │   ├── cli/          # 命令行接口
│   │   ├── eval/         # 评估工具
│   │   ├── jit_kernel/   # JIT编译内核
│   │   ├── lang/         # 前端语言接口
│   │   ├── multimodal_gen/ # 多模态生成
│   │   ├── srt/          # SGLang运行时核心
│   │   │   ├── batch_invariant_ops/   # 批处理不变操作
│   │   │   ├── entrypoints/           # 服务入口点
│   │   │   ├── managers/              # 管理器组件
│   │   │   ├── model_executor/        # 模型执行器
│   │   │   ├── models/                # 模型实现
│   │   │   ├── sampling/              # 采样算法
│   │   │   └── ...                    # 其他组件
│   │   ├── bench_*.py    # 基准测试脚本
│   │   ├── launch_server.py # 服务器启动脚本
│   │   └── ...           # 其他工具模块
├── sgl-kernel/            # SGLang专用内核库
├── sgl-model-gateway/     # 模型网关
├── test/                  # 测试代码
└── pyproject.toml         # 项目配置文件
```

## 核心逻辑讲解

### 1. 前端语言设计
SGLang提供了类似Python的高级编程接口，允许用户以函数式方式编写LLM应用：

```python
@function
def multi_turn_chat(s):
    s += system("You are a helpful assistant.")
    s += user("What is the capital of France?")
    s += assistant(gen(max_tokens=64))
    s += user("What is its population?")
    s += assistant(gen(max_tokens=64))
```

### 2. 请求处理流程
```
用户请求 → API服务器 → 分词器管理器 → 调度器 → 模型执行器 → 解分词器 → 响应返回
```

- **分词器管理器**: 负责将文本请求转换为token ID
- **调度器**: 管理请求队列，进行批处理调度
- **模型执行器**: 执行模型推理计算
- **解分词器**: 将生成的token ID转换为文本

### 3. 性能优化机制
- **RadixAttention**: 通过前缀缓存避免重复计算
- **连续批处理**: 动态批处理不同长度请求
- **推测解码**: 使用小模型加速大模型推理
- **分页注意力**: 高效管理长序列KV缓存

### 4. 分布式支持
- **张量并行**: 将模型权重切分到多个GPU
- **流水线并行**: 将模型层切分到多个GPU
- **专家并行**: 适用于MoE模型的专家分配

## 快速启动

### 环境准备
确保系统安装了Python 3.10+和CUDA环境（用于GPU加速）。

### 安装
```bash
# 安装SGLang
pip install sglang

# 或从源码安装
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e .
```

### 启动服务器
```bash
# 启动SGLang服务器
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

### 客户端使用
```python
from sglang import function, gen, set_default_backend, Runtime

# 设置后端
runtime = Runtime(endpoint="http://localhost:30000")
set_default_backend(runtime)

# 定义生成函数
@function
def generate_text(s, prompt):
    s += prompt
    s += gen("response", max_tokens=128)

# 执行生成
result = generate_text.run(prompt="Hello, how are you?")
print(result["response"])
```

## 扩展方向

### 短期规划
- 更多模型架构支持
- 新硬件平台适配
- 性能优化改进
- 更丰富的API功能

### 长期愿景
- 更智能的自动优化
- 更强的多模态能力
- 更完善的生态工具
- 企业级功能增强

## 联系我们

对于希望在大规模生产环境中采用或部署SGLang的企业，请联系我们：sglang@lmsys.org

---

SGLang项目是一个开放源码项目，欢迎社区贡献和反馈。如果您在使用过程中遇到问题或有改进建议，请在GitHub上提交Issue或Pull Request。