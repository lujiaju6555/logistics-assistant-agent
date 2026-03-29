# 基于 LangChain 的智能物流客服助手

## 项目简介

本项目是物流企业智能客服系统，采用模块化、流水线式架构，实现了完整的客服对话流程。

框架：感知->抽取->执行->风控

核心亮点：
 - 混合检索：集成 Qwen-Embedding + BM25 对原问题和重写问题进行多路召回，结合BGE-M3进行重排序，Recall@5 达到 70%
 - 上下文记忆：支持多轮对话，记录用户结构化信息，并在每轮对话时抽取最新的结构化信息

## 项目结构

```
logistics-assistant-agent/
├── config/            # 配置管理
│   ├── __init__.py
│   ├── settings.py    # 系统设置
│   ├── prompt.py      # 提示词管理
├── data/              # 数据目录
│   ├── database/      # 业务数据库（Excel文件）
│   │   ├── OMS.xlsx   # 订单管理系统数据
│   │   ├── TMS.xlsx   # 运输管理系统数据
│   │   └── WMS.xlsx   # 仓库管理系统数据
│   ├── knowledge_base/ # 知识库
│   ├── chat_history/  # 对话历史
│   ├── user_structured_info/ # 用户结构化信息
│   ├── golden_test_set.jsonl # 黄金测试集
│   └── test_input.json # 测试输入文件
├── modules/           # 核心模块
│   ├── perception/    # 感知层（意图识别、问题重写）
│   ├── extraction/    # 信息抽取层
│   ├── execution/     # 执行层（工具调用、RAG检索）
│   └── safety/        # 风控层
├── utils/             # 工具模块
│   ├── data_loader.py # 数据加载器
│   ├── memory_manager.py # 内存管理器
│   └── pipeline_state.py # 流水线状态
├── main_pipeline.py   # 主流水线
├── generate_data.py   # 黄金测试集生成
├── evaluate.py        # 评估脚本
├── requirements.txt   # 依赖管理
└── .env               # 环境变量
```

## 环境配置

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量（.env文件）：
   ```
   # 通义千问API密钥（用户需要自己设置）
   DASHSCOPE_API_KEY=your_api_key
   ```
   
## 完整流程

### 1. 数据生成
- **生成黄金测试集**：使用 `generate_data.py` 生成包含问题-答案-来源的黄金测试集
  ```bash
  python generate_data.py --output data/golden_test_set.jsonl --sample-size 100
  ```

### 2. 格式转换
- **转换为系统输入格式**：使用 `data/convert_test_set_to_input.py` 将黄金测试集转换为系统输入格式
  ```bash
  python data/convert_test_set_to_input.py --golden-file data/golden_test_set.jsonl --output-file data/test_input.json --users-count 5
  ```

### 3. 客服回答

```bash
python main_pipeline.py --input data/test_input.json --output result/output.json
```

系统处理用户问题的流程分为四个步骤：

#### 3.1 感知层
- **意图识别**：识别用户意图（闲聊、重置、转人工、业务查询(下分订单状态查询、物流状态查询、下单操作等子意图)）
- **问题重写**：解决指代消解问题，结合上下文补充，生成完整查询

#### 3.2 信息抽取层
- **信息抽取**：从问题中提取结构化参数
- **结构化记忆**：结合用户历史结构化信息，更新用户档案

#### 3.3 执行层
- **业务工具调用**：调用业务工具（查询物流状态、订单状态等）
- **RAG检索**：
  - 手动爬取京东物流官网服务细则数据
  - 基于qwen3-vl-flash将图片信息提取为文本格式
  - 基于LangChain进行文本切片
  - 基于Qwen-embedding嵌入模型构建向量数据库
  - 结合稀疏检索（BM25）进行多路召回（原问题+Dense、原问题+Sparse、重写问题+Dense、重写问题+Sparse）
  - 智能融合并使用BGE-m3重排序（可以下载到本地，需要适应性调整）

#### 3.4 风控层
- **风险检测**：检测回答中的风险内容（敏感信息、不当言论等）

### 4. 评估结果
- **运行评估**：使用 `evaluate.py` 评估系统性能
  ```bash
  python evaluate.py --input-file data/input.json --result-file result/output.json
  ```
- **评估指标**：recall、hit@k

## 技术栈

- Python 3.8+
- LangChain
- API（如Qwen-embedding）
- FAISS（向量数据库）
- Pydantic（数据验证）
- Pandas（Excel数据处理）
- Sentence Transformers（重排序模型）

## 注意事项

1. **API密钥**：系统使用通义千问API，需要在.env文件中配置有效的API密钥。
2. **知识库**：首次运行时会自动构建知识库，后续运行会使用已构建的向量存储。
3. **业务数据**：系统从data/database目录中的Excel文件读取业务数据，包括OMS.xlsx（订单管理）、TMS.xlsx（运输管理）和WMS.xlsx（仓库管理）。
4. **多轮对话**：系统支持多轮对话，会自动存储和加载用户的对话历史和结构化信息。
5. **评估**：使用黄金测试集评估系统性能，计算hit@k和recall指标。

## 后续改进

 - 数据生成的方式有待改进
 - 真值打标应使用大模型对问题和知识库中每个文档进行匹配判断，以完整地判断
 - 文本切片方式可以改变切片大小生成不同颗粒度的切片，从而增加检索能力，同时可以修改为父子检索方式
 - 探索更强的重排序模型（Qwen-Reranker）

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=lujiaju6555/logistics-assistant-agent&type=date&legend=top-left)](https://www.star-history.com/?repos=lujiaju6555%2Flogistics-assistant-agent&type=date&legend=top-left)
