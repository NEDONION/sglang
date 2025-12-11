# SGLang 核心流程与优化技术

## 1. 整体运行流程

### 1.1 架构流程图
```
用户请求
    ↓
客户端 API (lang/api.py)
    ↓
HTTP/gRPC 服务器 (entrypoints/http_server.py)
    ↓
请求解析和验证
    ↓
TokenizerManager (managers/tokenizer_manager.py)
    ↓
请求队列管理
    ↓
Scheduler (managers/scheduler.py)
    ↓
批处理调度
    ↓
ModelExecutor (model_executor/model_runner.py)
    ↓
硬件后端推理 (hardware_backend/)
    ↓
采样 (sampling/)
    ↓
Detokenizer (managers/detokenizer_manager.py)
    ↓
响应返回给用户
```

### 1.2 服务器启动流程 (launch_server.py)

#### 启动入口
```python
# launch_server.py
if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    run_server(server_args)
```

#### 服务器类型
- **HTTP模式**: 默认模式，使用FastAPI提供OpenAI API兼容接口
- **gRPC模式**: 高性能模式，适用于高吞吐量场景

### 1.3 Engine初始化流程 (entrypoints/engine.py)

#### Engine类结构
Engine是SGLang的核心入口点，由三个主要组件构成：

1. **TokenizerManager**: 处理请求分词和结果解分词
2. **Scheduler (子进程)**: 请求调度、批处理和模型推理调度
3. **DetokenizerManager (子进程)**: 将生成的tokenID转换为文本

#### 初始化步骤
1. 解析服务器参数
2. 配置硬件设备和多进程环境
3. 启动子进程 (Tokenizers, Scheduler, Detokenizers)
4. 建立ZMQ通信连接

## 2. 详细请求处理流程

### 2.1 请求处理流程

#### 2.1.1 请求接收和验证
- HTTP/gRPC服务器接收用户请求
- 验证请求参数格式
- 将请求转换为内部数据结构

#### 2.1.2 分词器处理
```python
# tokenizer_manager.py
# 1. 接收原始请求 (GenerateReqInput)
# 2. 文本分词为token ID
# 3. 多模态数据预处理
# 4. 将处理后请求发送给调度器
```

#### 2.1.3 调度流程 (scheduler.py)
主要调度组件:
- **请求队列管理**: 管理等待处理的请求
- **批处理策略**: 动态批处理多个请求
- **内存管理**: 管理KV缓存和注意力计算所需的内存
- **前缀缓存**: 使用RadixAttention优化重复提示的处理

##### 调度循环
1. **请求入队**: 新请求加入等待队列
2. **批处理构建**: 选择可执行的请求批次
3. **内存分配**: 为批次请求分配KV缓存
4. **模型推理**: 执行前向传播
5. **结果处理**: 处理模型输出，更新请求状态
6. **响应返回**: 将结果返回给分词器

#### 2.1.4 模型推理流程 (model_executor/model_runner.py)
##### 模型执行阶段
1. **输入准备**: 整理批次请求的输入数据
2. **前向传播**: 执行模型前向计算
   - 嵌入层
   - 多层Transformer计算
   - 注意力机制
   - 激活函数
3. **Logits处理**: 对模型输出进行处理
4. **采样**: 根据采样参数生成下一个token

##### 批处理类型
- **预填充(Prefill)**: 处理新请求的提示部分
- **解码(Decode)**: 生成新的输出token
- **混合批处理**: 同时处理预填充和解码请求

### 2.2 通信机制

#### 进程间通信 (ZMQ)
- **Tokenizers ↔ Scheduler**: 发送分词后的请求，接收模型输出
- **Scheduler ↔ Detokenizer**: 发送生成的token ID，接收解码文本
- **主进程 ↔ 子进程**: 控制命令和状态同步

#### gRPC通信
- 外部客户端与服务器的高性能通信
- 分布式部署中的节点间通信

## 3. 内存管理流程

### 3.1 KV缓存管理
- **分页注意力**: 将KV缓存组织为页面
- **前缀缓存**: 使用Radix树缓存共同前缀
- **内存池**: 预分配内存，减少分配开销

### 3.2 内存优化技术
- **动态批处理**: 根据内存占用动态调整批次大小
- **卸载机制**: 将不活跃的缓存卸载到CPU
- **垃圾回收**: 管理未使用的内存页面

## 4. 多模态处理流程

### 4.1 图像处理
1. 图像输入预处理
2. 视觉编码器处理
3. 特征与文本token对齐
4. 多模态模型推理

### 4.2 视频处理
1. 视频帧采样
2. 逐帧视觉特征提取
3. 时间维度建模
4. 集成到文本生成流程

## 5. 采样和生成流程 (sampling/)

### 5.1 采样类型
- **贪心采样**: 选择最高概率token
- **Top-k采样**: 从top-k概率的token中采样
- **Top-p (Nucleus) 采样**: 从累积概率>top-p的最小token集合中采样
- **温度采样**: 调整输出随机性
- **约束生成**: 结构化输出生成

### 5.2 采样步骤
1. 获取Logits
2. 应用采样策略
3. 生成下一个token
4. 更新KV缓存

## 6. 高级优化技术

### 6.1 RadixAttention前缀缓存

```
共享前缀的Radix树结构:

                Root
                 │
            [system prompt]
                 │
        ┌────────┼────────┐
        │        │        │
   [user1: hi] [user2: hi] [user3: hello]
        │        │        │
   [ass: hello] [ass: hi] [ass: greetings]
        │        │        │
    [user: how] [user: how] [user: how are]
        │        │        │
     [ass: are] [ass: are] [ass: you doing]

每个节点存储对应的KV缓存，相同前缀无需重复计算
```

### 6.2 连续批处理 vs 传统批处理

```
传统批处理 (Fixed batches):
Batch 1: [Req1, Req2, Req3] - 等待最长请求完成
Batch 2: [Req4, Req5, Req6] - 等待最长请求完成

连续批处理 (Continuous batching):
Time: 0  1  2  3  4  5  6  7  8  9  10 11 12
     [R1,R2,R3]              # 3个请求同时开始
     [R1,R2,R3]              # R1在时刻4完成
        [R2,R3,R4]           # R4立即加入，无需等待R2,R3
        [R2,R3,R4]           # R2在时刻6完成
           [R3,R4,R5]        # R5立即加入
              [R4,R5]        # R3完成
              [R4,R5]        # R4完成
                 [R5,R6]     # R6加入
                    [R6]     # R5完成
                    [R6]     # R6完成
```

### 6.3 推测解码
- 使用草稿模型快速生成候选token
- 通过目标模型验证候选token
- 加速生成速度

### 6.4 分页注意力
- 将KV缓存分页管理
- 支持更长的序列处理

### 6.5 CUDA图优化
- 减少Kernel启动开销
- 对固定形状的计算进行优化

### 6.6 张量并行处理流程

```
4-GPU张量并行示例 (模型权重分割):

GPU 0: [W_00, W_01, W_02, W_03]
GPU 1: [W_10, W_11, W_12, W_13]
GPU 2: [W_20, W_21, W_22, W_23]
GPU 3: [W_30, W_31, W_32, W_33]

前向传播过程:
Input ──► [GPU 0: W_00·Input, W_01·Input, W_02·Input, W_03·Input]
          [GPU 1: W_10·Input, W_11·Input, W_12·Input, W_13·Input]
          [GPU 2: W_20·Input, W_21·Input, W_22·Input, W_23·Input]
          [GPU 3: W_30·Input, W_31·Input, W_32·Input, W_33·Input]

All-Reduce: 各GPU计算结果相加得到最终输出
```

## 7. 性能优化技术详解

### 7.1 分页注意力内存管理

```
KV缓存页面管理:

物理内存 (Physical Memory):
┌─────────────────────────────────────────────────────────┐
│ Page 0 │ Page 1 │ Page 2 │ Page 3 │ ... │ Page N-1     │
│[KV_Cache] [KV_Cache] [KV_Cache] [KV_Cache] [KV_Cache]   │
└─────────────────────────────────────────────────────────┘

逻辑序列 (Logical Sequences):
┌─────────┐    ┌─────────────────┐    ┌─────┐
│ Seq 1:  │    │ Seq 2:          │    │ Seq 3: │
│ P0, P1  │    │ P2, P4, P7      │    │ P3   │
│[tok0-15]│    │[tok0-15,16-31,  │    │[tok0-│
│[tok16-31]│   │ 32-47]         │    │ 15]  │
└─────────┘    └─────────────────┘    └─────┘
```

### 7.2 多级缓存策略

```
缓存层级结构:

L1: GPU显存缓存 (KV Cache)
    ├── 活跃请求的KV状态
    ├── 通过分页机制管理
    └── 快速访问，高带宽

L2: Radix前缀缓存
    ├── 共享前缀的KV状态
    ├── 持久化存储，避免重复计算
    └── 高效的树结构管理

L3: CPU内存缓存 (可选)
    ├── 非活跃请求的KV状态
    ├── 通过卸载机制管理
    └── 低优先级，节省GPU内存
```

### 7.3 分布式部署流程

#### 7.3.1 张量并行
- 模型权重在多个GPU间分割
- 并行计算模型输出
- 使用AllReduce同步结果

#### 7.3.2 流水线并行
- 模型层在多个GPU间分割
- 重叠计算和通信
- 处理长序列模型

#### 7.3.3 专家并行 (MoE)
- MoE模型的专家在GPU间分配
- 门控机制决定路由
- 高效处理稀疏激活

## 8. 前端语言执行流程

### 8.1 函数定义
1. 使用`@function`装饰器定义SGL函数
2. 生成中间表示(IR)
3. 编译为可执行图

### 8.2 程序执行
1. 解释器执行IR
2. 调用后端API
3. 处理响应和状态

### 8.3 高级功能
- **控制流**: 支持条件判断和循环
- **外部交互**: 调用外部API或工具
- **并行生成**: 同时生成多个输出

### 8.4 前端语言使用模式

```python
# 模式1: 简单生成
@function
def simple_gen(s, prompt):
    s += prompt
    s += gen("response", max_tokens=100)

# 模式2: 对话流程
@function
def chat_dialogue(s, user_input):
    s += system("You are a helpful assistant.")
    s += user(user_input)
    s += assistant(gen(max_tokens=128))

# 模式3: 条件生成
@function
def conditional_gen(s, question):
    s += f"Question: {question}\nAnswer:"
    # 基于第一个词决定生成策略
    first_word = gen(max_tokens=1, choices=["Yes", "No"])
    if first_word == "Yes":
        s += gen(max_tokens=50)
    else:
        s += gen(max_tokens=20)
```

## 9. 性能监控和指标收集

### 9.1 指标类型
- 请求延迟
- 吞吐量
- GPU利用率
- 内存使用率
- 缓存命中率

### 9.2 监控流程
1. 收集各组件性能数据
2. 聚合并计算指标
3. 导出到监控系统
4. 提供API查询接口

## 10. 配置优化示例

### 10.1 高吞吐量配置
```python
# 高吞吐量配置
ServerArgs(
    chunked_prefill_size=2048,  # 启用分块预填充
    max_running_requests=1000,   # 高并发
    enable_overlap_schedule=True, # 重叠调度
    attention_backend="flashinfer" # 高效注意力
)
```

### 10.2 低延迟配置
```python
# 低延迟配置
ServerArgs(
    stream_output=True,          # 流式输出
    schedule_policy="fcfs",      # 先到先服务
    radix_cache=True,            # 启用前缀缓存
    cuda_graph=False            # 关闭CUDA图（降低延迟）
)
```

这个流程展示了SGLang如何高效地处理LLM推理请求，通过多个优化技术和组件协作，实现了高性能的服务能力。