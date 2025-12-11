# SGLang 项目架构概览

## 1. 项目概述

SGLang (SGLang Runtime) 是一个高性能的大型语言模型（LLM）和视觉语言模型（VLM）服务框架。该项目旨在为大规模生产环境提供低延迟、高吞吐量的推理服务，支持从单GPU到分布式集群的各种部署场景。

### 1.1 核心设计思想
- **分层架构**: 前端语言层、运行时引擎层、硬件抽象层
- **模块化和可扩展**: 通过插件化架构支持多种模型、硬件后端和优化策略
- **性能优先**: 连续批处理、前缀缓存、推测解码等先进优化技术
- **易用性**: 提供直观的前端语言，简化LLM应用开发

### 1.2 技术栈
- **编程语言**: Python + PyTorch
- **核心框架**: Transformers、FlashInfer
- **网络通信**: FastAPI、aiohttp、gRPC
- **并发处理**: asyncio、uvloop
- **数据处理**: numpy、tiktoken、sentencepiece

## 2. 项目结构

### 2.1 根目录结构
```
sglang/
├── 3rdparty/          # 第三方依赖库
├── assets/            # 静态资源
├── benchmark/         # 基准测试
├── docker/            # Docker配置
├── docs/              # 文档
├── examples/          # 使用示例
├── notes/             # 项目笔记
├── python/            # Python源代码 (主要)
├── scripts/           # 脚本文件
├── sgl-kernel/        # SGLang内核库
├── sgl-model-gateway/ # 模型网关
├── test/              # 测试文件
└── pyproject.toml     # 项目依赖配置
```

### 2.2 Python源代码结构
```
sglang/python/sglang/
├── cli/               # 命令行接口
├── eval/              # 评估工具
├── jit_kernel/        # JIT编译内核
├── lang/              # 前端语言接口
├── multimodal_gen/    # 多模态生成
├── srt/               # SGLang运行时 (SGLang Runtime) - 核心
├── test/              # 测试
├── bench_*.py         # 基准测试脚本
├── check_env.py       # 环境检查
├── global_config.py   # 全局配置
├── launch_server.py   # 服务器启动脚本
├── profiler.py        # 性能分析器
├── utils.py           # 工具函数
└── version.py         # 版本信息
```

### 2.3 核心模块 - SGLang运行时 (srt/)
```
srt/
├── batch_invariant_ops/   # 批处理不变操作
├── batch_overlap/         # 批处理重叠
├── checkpoint_engine/     # 检查点引擎
├── compilation/           # 编译优化
├── configs/               # 配置文件
├── connector/             # 连接器
├── constrained/           # 约束生成
├── debug_utils/           # 调试工具
├── disaggregation/        # 解耦合
├── distributed/           # 分布式
├── dllm/                  # 扩散LLM
├── entrypoints/           # 入口点 (HTTP/gRPC服务器)
├── eplb/                  # 专家负载均衡
├── function_call/         # 函数调用
├── grpc/                  # gRPC相关
├── hardware_backend/      # 硬件后端
├── layers/                # 模型层实现
├── lora/                  # LoRA支持
├── managers/              # 管理器组件
├── mem_cache/             # 内存缓存
├── metrics/               # 指标收集
├── model_executor/        # 模型执行器
├── model_loader/          # 模型加载器
├── models/                # 模型实现
├── multimodal/            # 多模态
├── multiplex/             # 多路复用
├── parser/                # 解析器
├── sampling/              # 采样算法
├── speculative/           # 推测解码
├── tokenizer/             # 分词器
├── tracing/               # 追踪功能
├── utils/                 # 工具函数
├── weight_sync/           # 权重同步
├── constants.py           # 常量定义
├── custom_op.py           # 自定义操作
├── environ.py             # 环境变量
├── server_args.py         # 服务器参数
├── server_args_config_parser.py  # 服务器参数配置解析器
```

### 2.4 前端语言模块 (lang/)
```
lang/
├── backend/              # 后端实现
├── api.py                # 公共API
├── chat_template.py      # 聊天模板
├── choices.py            # 选择器
├── interpreter.py        # 解释器
├── ir.py                 # 中间表示
├── tracer.py             # 追踪器
```

## 3. 整体架构

### 3.1 核心架构图 (ASCII艺术)
```
                    ┌─────────────────────────────────────────┐
                    │              用户请求                    │
                    │           (HTTP/gRPC)                   │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │            API 服务器层                  │
                    │     (entrypoints/http_server.py)        │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │           分词器管理器                   │
                    │    (managers/tokenizer_manager.py)      │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │              调度器                     │
                    │      (managers/scheduler.py)            │
                    │  ┌─────────────┐ ┌──────────────────┐   │
                    │  │ 请求队列    │ │ 内存管理器       │   │
                    │  │ (Queue)     │ │ (MemoryManager)  │   │
                    │  └─────────────┘ └──────────────────┘   │
                    │  ┌─────────────┐ ┌──────────────────┐   │
                    │  │ 批处理器    │ │ KV缓存管理       │   │
                    │  │ (Batcher)   │ │ (KV Cache)       │   │
                    │  └─────────────┘ └──────────────────┘   │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │           模型执行器                     │
                    │   (model_executor/model_runner.py)      │
                    │  ┌─────────────┐ ┌──────────────────┐   │
                    │  │ 注意力层   │ │ 采样器          │   │
                    │  │ (Attention) │ │ (Sampler)        │   │
                    │  └─────────────┘ └──────────────────┘   │
                    │  ┌─────────────┐ ┌──────────────────┐   │
                    │  │ 前向传播   │ │ 量化支持        │   │
                    │  │ (Forward)   │ │ (Quantization)   │   │
                    │  └─────────────┘ └──────────────────┘   │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │          硬件后端层                     │
                    │    (hardware_backend/)                  │
                    │  ┌─────────────┐ ┌──────────────────┐   │
                    │  │ CUDA后端   │ │ ROCm后端        │   │
                    │  │ (CUDA)      │ │ (ROCm)           │   │
                    │  └─────────────┘ └──────────────────┘   │
                    │  ┌─────────────┐ ┌──────────────────┐   │
                    │  │ TPU后端    │ │ CPU后端         │   │
                    │  │ (TPU)       │ │ (CPU)            │   │
                    │  └─────────────┘ └──────────────────┘   │
                    └─────────────────────────────────────────┘
```

### 3.2 进程通信架构 (ZMQ)
```
主进程 (Engine)
    │
    ├─► Tokenizer Manager 进程 (ZMQ PUSH/PULL)
    │
    ├─► Scheduler 进程 (ZMQ PUSH/PULL)
    │   │
    │   ├─► 模型执行器进程 (TP Worker)
    │   │   │
    │   │   └─► GPU设备
    │
    └─► Detokenizer Manager 进程 (ZMQ PUSH/PULL)
```

### 3.3 前端语言架构
```
SGLang前端语言
├── api.py                 # 公共API定义
│   ├── gen()              # 生成函数
│   ├── select()           # 选择函数
│   ├── image(), video()   # 多模态输入
│   └── [其他语言构造]
│
├── ir.py (中间表示)
│   ├── SglFunction        # 函数表示
│   ├── SglGen            # 生成节点
│   ├── SglSelect         # 选择节点
│   ├── SglImage, SglVideo # 多模态节点
│   └── SglRoleBegin/End   # 角色节点
│
├── interpreter.py        # 语言解释器
│   ├── ProgramState      # 程序状态
│   └── Interpreter       # 执行逻辑
│
├── backend/ (后端实现)
│   ├── base_backend.py   # 后端基类
│   ├── runtime_endpoint.py # 运行时后端
│   └── [其他后端实现]
│
└── tracer.py            # 追踪器 (用于预缓存)
```

## 4. 功能特性

### 4.1 主要功能特性
- **高速后端运行时**: RadixAttention前缀缓存、零开销CPU调度器、预填充-解码解耦
- **模型并行**: 张量并行、流水线并行、专家并行、数据并行
- **结构化输出**: JSON模式和有限状态机的快速结构化输出生成
- **量化支持**: FP4/FP8/INT4/AWQ/GPTQ等多种量化格式
- **多模态支持**: 图像、视频等多模态输入处理

### 4.2 硬件支持
- **NVIDIA GPU**: GB200/B300/H100/A100/Spark等
- **AMD GPU**: MI355/MI300等
- **Intel平台**: Xeon CPU、XPU等
- **Google TPU**: TPU v4/v5等
- **Ascend NPU**: 华为昇腾NPU

### 4.3 模型支持
- **LLM模型**: Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral等
- **多模态模型**: LLaVA、Qwen-VL、InternVL、Idefics2等
- **专业模型**: 嵌入模型、奖励模型、分类模型

### 4.4 通信协议支持
- **OpenAI API兼容**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` 等
- **gRPC接口**: 提供高性能的gRPC服务接口，适用于高吞吐量场景

## 5. 部署架构模式

### 5.1 单机部署模式
```
┌─────────────────────────────────────────┐
│                单机SGLang                │
│  ┌─────────────┐ ┌──────────────────┐   │
│  │ HTTP Server │ │ GPU Workers      │   │
│  │ (多个端口)   │ │ (TP=4)          │   │
│  └─────────────┘ └──────────────────┘   │
│         │                    │          │
│         └────────────────────┘          │
│              共享GPU内存                  │
└─────────────────────────────────────────┘
```

### 5.2 分布式部署模式
```
┌─────────────────┐    ┌─────────────────┐
│   控制节点       │    │   工作节点      │
│  ┌─────────────┐│    │┌──────────────┐│
│  │ API Server  ││    ││ GPU Workers  ││
│  │ (负载均衡)   ││────││ (TP=8)      ││
│  └─────────────┘│    │└──────────────┘│
│                 │    │  共享存储      │
└─────────────────┘    └─────────────────┘
```

### 5.3 与同类框架对比
| 特性 | SGLang | vLLM | TGI | TensorRT-LLM |
|------|--------|------|-----|--------------|
| RadixAttention | ✅ | ❌ | ❌ | ❌ |
| 连续批处理 | ✅ | ✅ | ❌ | ❌ |
| 预填充-解码解耦 | ✅ | ❌ | ❌ | ❌ |
| 规则约束生成 | ✅ | ❌ | ✅ | ❌ |
| 多模态支持 | ✅ | 部分 | 部分 | ❌ |
| 模型切换速度 | 快 | 中 | 快 | 慢 |
| 内存效率 | 高 | 高 | 中 | 高 |

SGLang的独特优势包括RadixAttention、连续批处理、预填充-解码解耦、推测解码、结构化输出等。