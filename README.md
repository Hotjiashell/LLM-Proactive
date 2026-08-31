# LLM-Proactive

面向任务型与开放域对话的提示词实验代码。项目比较三类提示策略：直接生成回复、先做与任务相关的中间决策再回复（Proactive），以及先分析再决策再回复（ProCoT）。它覆盖四类离线基准和一个主题引导的自博弈任务：歧义问答、文档问答、话题过渡、价格协商与目标话题到达。

这是一份 2023 年的研究复现实验代码，而不是封装好的 Python 包。它没有依赖锁定、命令行统一入口、CI 或许可证文件；多个脚本还保留了硬编码文件名、模型路径与旧版 OpenAI 客户端调用。本文档以仓库当前代码为准，明确哪些命令可直接执行，哪些地方需要先调整。

## 方法概览

每个样本会按不同方法构造多条 prompt，并让同一模型分别生成：

| 方法键 | 含义 |
| --- | --- |
| `zs_standard` / `fs_standard`，或 `zs_resp` / `fs_resp` | 零样本 / 少样本的直接回复基线。键名随任务而变化。 |
| `zs` / `fs` | 零样本 / 少样本 Proactive。模型先显式预测歧义、桥接话题、协商策略或对话行为等中间变量，再生成回复。 |
| `zs-pcot` / `fs-pcot`，或 `zs_pcot` / `fs_pcot` | ProCoT 变体。要求先给出对当前状态的分析，再输出中间决策和回复。 |
| `cqg`、`fs_cqg`、`zs_pcot_cqg`、`fs_pcot_cqg` | 仅在歧义问答样本中生成澄清问题的变体。 |

问答任务的 Proactive 输出必须遵从 prompt 指定的文本模板，例如以 `The answer is` 或 `The clarifying question is` 开头。评测器依赖这些精确短语解析结果，改写 prompt 或使用不遵从模板的模型时，评测结果会失真。

## 任务与目录

| 目录 | 任务 | 原始测试数据 | 处理结果 | 主要指标 |
| --- | --- | --- | --- | --- |
| `abgcoqa/` | 基于文档与历史对话的歧义问答 | `abg-coqa-test.txt` | `abgcoqa-source.txt`、`abgcoqa-target.txt` | 澄清需求预测（CNP）的 Precision / Recall / F1，澄清问题生成（CQG）的 BLEU-1..4 |
| `pacific/` | 表格和段落支撑的歧义问答 | `validation.json` | `pacific-source.txt`、`pacific-target.txt` | CNP 的 Precision / Recall / F1，CQG 的 ROUGE-1 / 2 / L |
| `otters/` | 以桥接话题逐步接近目标话题的开放域对话 | `otters-test.json` | `otters-source.txt`、`otters-target.txt` | BLEU、METEOR、ROUGE-L、CIDEr，以及话题命中率 Hit@1 / Hit@3 |
| `negotiate/` | 卖方与买方的价格协商 | `negotiate_data.pkl.zip` | `negotiate-source.txt`、`negotiate-target.txt` | 回复 BLEU / BERTScore，协商策略和对话行为的 F1 / AUC |
| `topkg/` | 双角色自博弈，将对话自然地引向目标话题 | `topkg-test.json` | `output/*.json` | 成功率、成功时的轮数、相邻轮次语义相似度 |

根目录的 `data/` 保存了 `abgcoqa`、`otters`、`pacific` 和 `topkg` 的同一份输入副本。当前处理脚本读取各任务目录中的文件，而不是 `data/` 下的副本；两处数据的 SHA-256 已核对一致。`negotiate` 的压缩数据必须先解压。

```text
LLM-Proactive/
├── abgcoqa/       # 歧义 CoQA：数据处理、ChatGPT/Vicuna 推理、评测
├── otters/        # 话题过渡：数据处理、ChatGPT/Vicuna 推理、评测
├── negotiate/     # 协商：数据处理、ChatGPT/Vicuna 推理、评测
├── pacific/       # 表格/段落问答：数据处理、ChatGPT/Vicuna 推理、评测
├── topkg/         # 目标话题引导的 ChatGPT/Vicuna 自博弈与评测
├── data/          # 输入数据副本
└── README.md
```

## 环境准备

仓库未提供经验证的版本组合。建议用独立的 Python 3.9 或 3.10 环境，并按准备运行的任务安装依赖。下面是由 import 语句整理的最小起点，不表示经过完整的版本锁定复现：

```bash
git clone https://github.com/dengyang17/LLM-Proactive.git
cd LLM-Proactive
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers fschat 'openai<1' numpy scikit-learn nltk tqdm bert-score
python -m pip install sacrebleu pycocoevalcap spacy pytextrank pyrouge
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4
```

说明：

- `fschat` 提供 `fastchat.model.load_model`，用于本地 Vicuna 推理。
- ChatGPT 脚本使用已经废弃的 `openai.ChatCompletion.create` 调用方式和 `gpt-3.5-turbo-0301` 默认模型。因此它们不能和 `openai>=1` 直接搭配；即使固定在旧客户端版本，也应把模型名换成账户当前可用的模型。长期使用时建议迁移至新版 OpenAI SDK。
- 本地 Vicuna 与 TopKG 的语义相似度计算都假定 CUDA 可用。TopKG 会加载 `sentence-transformers/all-MiniLM-L6-v2`，但其 tokenizer 和模型的 `cache_dir` 仍含作者机器路径或 `your_path_to/plm` 占位符。运行 `topkg/self_play.py`、`topkg/chatgpt_self_play.py` 或 `topkg/eval.py` 前，请把这些 `cache_dir` 改为可写目录，或删除该参数以使用 Hugging Face 默认缓存。
- `pacific/eval.py` 的 CQG 自动评测通过 `pyrouge` 调用外部 ROUGE-1.5.5。安装 Python 包后，还需安装 ROUGE 并按 `pyrouge` 的说明设置其路径。
- `otters/eval.py` 的话题命中率会加载 spaCy 的 `en_core_web_sm` 模型并注册 `pytextrank` 管线。

## 数据预处理

所有路径都相对于当前工作目录。因此请先进入目标任务目录，再运行脚本。四个 `process.py` 在模块导入时就会执行，不需要再传子命令。

```bash
cd abgcoqa
python process.py
```

```bash
cd otters
python process.py
```

```bash
cd pacific
python process.py
```

```bash
cd negotiate
unzip negotiate_data.pkl.zip
python process.py
```

每行生成的 `*-source.txt` 和 `*-target.txt` 都是 Python 字典的字符串表示，不是 JSON。推理与评测脚本用 `eval()` 读取它们，所以只能处理本仓库可信产生的文件，不能将不可信输入交给这些脚本。

## 运行模型

### OpenAI Chat Completions 脚本

`abgcoqa/test_chatgpt.py`、`otters/otters_chatgpt.py`、`pacific/test_chatgpt.py`、`negotiate/negotiate_chatgpt.py` 和 `topkg/chatgpt_self_play.py` 是旧版 Chat Completions 运行器。ABG-CoQA、PACIFIC 与协商运行器会遍历一行 source dict 中的所有方法键，并将同一行的输出写回一个 dict，因此可填补已有行中缺少的键。OTTERS 的已有输出分支却只重跑 `zs_resp` 和 `fs_resp`，不会补全其余键；对 OTTERS 不要依赖断点续跑，除非先修正该脚本。

不要将 API Key 写进源代码。各脚本已导入 `os`，建议把其中的 `API_KEY = ...` 或函数内赋值替换为：

```python
API_KEY = os.environ["OPENAI_API_KEY"]
```

然后在 shell 中设置凭据并从对应目录运行：

```bash
export OPENAI_API_KEY="..."
cd abgcoqa
python test_chatgpt.py
```

其他三个单轮任务分别运行 `otters_chatgpt.py`、`test_chatgpt.py`（位于 `pacific/`）和 `negotiate_chatgpt.py`。请先检查脚本末尾的输入/输出相对路径：协商脚本目前写的是 `data/negotiate-source.txt` 和 `output/negotiate-chatgpt.txt`，与 `negotiate/process.py` 实际生成的位置不同，需要统一后再运行。若保留默认输出路径，还要先执行 `mkdir -p output`。

TopKG 会执行 12 组方法与目标难度的自博弈，且要求输出目录存在：

```bash
cd topkg
mkdir -p output
python chatgpt_self_play.py
```

### 本地 Vicuna 脚本

`abgcoqa/test_llama.py`、`otters/otters_vicuna.py`、`negotiate/negotiate_vicuna.py`、`pacific/test_llama.py` 和 `topkg/self_play.py` 使用 FastChat 加载本地 Hugging Face 格式的 Vicuna 模型。它们虽然注册了 FastChat 的模型参数，但随后会在源文件末尾覆盖 `args.model_path`：

```python
args.model_path = "your_path_to/vicuna_hf/13B"
```

先将该占位路径改为本地模型目录，然后从任务目录运行。例如：

```bash
cd abgcoqa
python test_llama.py --device cuda --num-gpus 1
```

这会生成 `abgcoqa-vicuna-13B.txt`。其他任务脚本的输出文件名可在各自 `infer(...)` 调用处确认。推理代码将输入张量直接迁移到 CUDA，CPU-only 环境不可用。

## 评测

评测脚本在 `__main__` 中写死了作者当时的结果文件名，`otters/eval.py` 还引用了当前仓库中不存在的 `../data/otters-target.txt`。更可靠的做法是在任务目录中按函数签名调用并传入本次生成的文件：

| 任务 | 可调用函数 |
| --- | --- |
| `abgcoqa/` | `evaluate(output_file, target_file)` |
| `pacific/` | `evaluate(output_file, target_file)` |
| `otters/` | `evaluate(output_file, target_file)` |
| `negotiate/` | `evaluate(output_file, target_file, response_file)` |
| `topkg/` | `eval_topkg(output_file)` |

以 ABG-CoQA 为例，在 `abgcoqa/` 下把 `eval.py` 最后的调用改为当前输出文件和 `abgcoqa-target.txt`，再执行：

```bash
python eval.py
```

PACIFIC 和 OTTERS 可采用相同方式，并分别使用 `pacific-target.txt`、`otters-target.txt`。PACIFIC 的 CQG 评测会创建 `tmp/reference/` 与 `tmp/candidate/`，但不会创建父目录，因此请先执行 `mkdir -p tmp`；OTTERS 会创建 `output.txt`；协商评测会创建 `bert_score.txt` 和传入的回复文件。这些均是运行产物。

TopKG 可以直接在 Python 中调用 `eval_topkg`，或修改 `topkg/eval.py` 最后的示例文件名，使其指向 `output/` 下实际产生的 ChatGPT 自博弈结果。该评测返回 `[success_rate, average_turns, coherence]`；成功率和轮数遵循脚本内部的布尔标记语义。`topkg/self_play.py` 的本地 Vicuna 路径只写入 `dialog` 和 `target`，而评测器还需要 `succ` 与 `turns`，所以本地结果必须先补写这两个字段或修改评测器，不能直接传给 `eval_topkg`。

### 协商任务的已知不匹配

当前 `negotiate/process.py` 只构造 `zs_resp`、`fs_resp`、`zs`、`fs`、`zs_pcot` 与 `fs_pcot` 六类 prompt。但 `negotiate/eval.py` 还读取 `zs_strat`、`fs_strat`、`zs_act`、`fs_act` 四个输出键，以评测标准方法的策略与对话行为。这四类输入并不会由现有预处理脚本生成，所以直接对默认生成结果运行评测会因缺少键而失败。复现协商任务前，需要补齐对应 prompt/推理输出，或让评测器只比较现有的 Proactive / ProCoT 键。

## 提示词与数据格式

- 每个任务目录的 `prompt.txt` 逐行定义方法指令和 few-shot demonstration；处理脚本按固定行号读取，不能随意增删或换序。
- 文档问答任务会按空格截断 prompt 的末尾 512 个 token（这是近似的空格切分，不是模型 tokenizer 的真实 token 数）。
- 推理输出是逐行 Python dict。不同任务的 PCoT 键既有连字符形式（`zs-pcot`）也有下划线形式（`zs_pcot`）；下游代码必须沿用本任务的键名。
- ABG-CoQA、PACIFIC 和协商运行器会把已有输出文件当作断点续跑状态，并以写模式重写同一文件、保留已有行后填补缺少的键。OTTERS 不具备这一行为，详见上文。无论任务类型，都不要混用来自不同数据顺序的输出文件。

## 已知限制与安全注意事项

- 仓库中没有 `requirements.txt`、版本锁定、随机种子控制或端到端测试，因此结果不能被视为严格的可复现基线。
- 所有数据处理、推理与评测路径依赖当前工作目录；从仓库根目录直接运行大多数脚本会找不到数据或 prompt。
- ChatGPT 运行器使用过时 SDK / 模型设置；新项目应迁移到当前的客户端接口，并将模型、重试、速率限制和输出目录配置化。
- 安全告警：`negotiate/negotiate_chatgpt.py` 当前含有一个已提交的 API 凭据，不能直接使用或传播。该凭据应立即在服务端轮换 / 吊销，并从 Git 历史与源文件中移除；之后只从环境变量或密钥管理系统读取新凭据。
- 仓库没有提供许可证或推荐引用信息。若要分发、二次使用或在论文中引用，请先联系原作者或补充相应的 `LICENSE` / `CITATION.cff`。

## 贡献建议

在扩展实验前，优先补齐以下基础设施：统一的 `requirements.txt` 或 lockfile、参数化的 CLI、JSONL 替代 `eval()` 序列化、环境变量凭据读取、可配置的模型与输出目录，以及覆盖每个数据处理器和评测器的最小测试。这样既能减少路径错误，也能让后续模型和 prompt 对比有可追溯性。

## ClarQ 多轮检索评测

`clarq/` 将 LLM-Proactive 的“先作中间决策、再执行动作”策略接入 Huawei ClarQ。它直接复用 `../huawei_dial/workspace/eval` 的 ClarQ 测试数据、混合检索器、grounded / random 用户模拟器、Success Judge、轨迹指标和报告；不会复制或改变这些评测口径。

策略模型每回合只调用一次：先以自由自然语言完成 ProCoT 分析，再在同一响应中调用 Huawei 原生的 `clarify_user` 或 `search_case` 工具；完成时则让最后一行输出 `Complete`。适配器直接转发 Huawei 的同一份 `TOOLS` 定义，记录分析文本后再规范化结果，因此策略服务需要支持 OpenAI-compatible function calling。每条非基础设施失败的轨迹都会额外保存 `proactive_policy.decisions`，而 `run_config.json` 会记录适配器版本，避免与非 Proactive 轨迹混合断点续跑。

```bash
cd clarq
python3 -m pip install -r ../../huawei_dial/workspace/eval/requirements.txt
cp config.example.env .env
# 编辑 .env 中的模型、用户模拟器、Elasticsearch 和 embedding 服务配置
bash run_evaluation.sh --check-only
bash run_evaluation.sh --limit 20 --output-dir ../../huawei_dial/workspace/eval/outputs/llm-proactive-smoke
```

完整配置、评测参数与离线测试见 [clarq/README.md](clarq/README.md)。
