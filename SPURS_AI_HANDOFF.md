# SPURS 部署与脚本开发交接

记录日期：2026-09-04。面向后续编写蛋白稳定性预测脚本的 AI。

## 1. 任务和当前状态

用户项目：`Thermal Stability Optimization.md`，目标蛋白 PDB 为 **1CXI，链 A**。计划用 SPURS 进行单点筛选、双点兼容性评估及最多 8 点的组合搜索，并结合保护位点、Rosetta、FoldX 等筛选。

**不要把部署进展误写为预测已全部成功。** 当前对话中的实际证据如下：

| 项目 | 状态 |
|---|---|
| RTX 5090 的 PyTorch GPU 计算测试 | 用户报告通过 |
| 模型文件下载、上传和解压 | 已完成，服务器日志确认 |
| 单点、多点及 ESM 离线缓存检查 | `Offline cache: PASS` |
| 单点模型初始化和权重加载 | 已执行到随后调用 PDB 解析的阶段 |
| 官方示例 PDB 解析 | 最后一次日志因缺少 `lmdb` 中断 |
| 补装 `lmdb`、`atom3d`、`tqdm` | 已给出命令，用户尚未反馈结果 |
| 完整单点/多点 GPU 推理验收 | 尚无通过日志，不可声称已通过 |
| 1CXI 实际预测、编号映射、候选筛选 | 本对话尚未执行 |
| 自动加载 `offline-env.sh` | 已给出 `.bashrc` 设置命令，执行情况未确认 |

下一步先解决依赖并跑通已有 `verify_spurs.py`，再开发生产脚本。不要因缓存或依赖问题重新下载所有模型或重装 GPU 环境。

## 2. 服务器环境

| 项目 | 已知信息 |
|---|---|
| 操作系统 | Ubuntu，具体发行版未核实 |
| 用户 | `root`，根据日志中的路径 |
| GPU | NVIDIA GeForce RTX 5090 |
| PyTorch | **2.8.0+cu128**，用户运行日志确认 |
| PyTorch CUDA runtime | **12.8**，用户运行日志确认 |
| 系统 CUDA | 用户最初描述为 CUDA 13.0；未独立核实 Toolkit/驱动详情 |
| Python | 用户选择的镜像标称 Python 3.12；精确版本需 `python -V` 确认 |
| SPURS 源码目录 | `/root/software/SPURS` |
| 分支 | 按教程安装 `fix/py311-inference-beta`；运行前核实 git 状态 |
| Git 提交 | 尚未从服务器读取；不能把模型仓库 revision 当作代码 commit |
| 网络 | 访问 Hugging Face 曾返回 `Network is unreachable` |

沿用当前 `python` 对应的环境，不要假定存在名为 `spurs5090` 的 Conda 环境。早期 Python 3.11/PyTorch 2.9.0 建议已被用户的镜像限制替代。

官方兼容分支面向 Python 3.11+ 推理，仍处于 beta。旧 `main` 教程的 Python 3.7/PyTorch 1.12/cu113 不适用于此部署。参见 [兼容分支 README](https://github.com/luo-group/SPURS/tree/fix/py311-inference-beta) 和 [PyTorch Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。

## 3. 模型文件与离线布局

模型仓库：`cyclization9/SPURS`。

已下载的模型仓库 revision：

```text
0cc7a565af8f31eb122819f95a9d16e27b3d1596
```

服务器目录：

```text
/root/models/spurs-offline/
├── hf_hub/
│   └── models--cyclization9--SPURS/
│       ├── refs/main
│       └── snapshots/0cc7a565af8f31eb122819f95a9d16e27b3d1596/
│           ├── spurs/
│           │   ├── .hydra/config.yaml
│           │   └── checkpoints/best.ckpt
│           └── spurs_multi/
│               ├── .hydra/config.yaml
│               └── checkpoints/best.ckpt
└── torch/hub/checkpoints/
    ├── esm2_t33_650M_UR50D.pt
    └── esm2_t33_650M_UR50D-contact-regression.pt
```

本地已实际检查的文件大小：

| 文件 | 字节数 |
|---|---:|
| 单点 `spurs/checkpoints/best.ckpt` | 2,678,927,823 |
| 多点 `spurs_multi/checkpoints/best.ckpt` | 2,697,880,281 |
| ESM 基础权重 | 2,604,537,549 |
| ESM contact-regression 权重 | 3,687 |
| 单点 config.yaml | 2,391 |
| 多点 config.yaml | 2,566 |
| `spurs-offline.tar` | 7,981,404,160 |

tar 共 8 个普通文件，包含上述 6 个文件、`refs/main` 和 `CACHEDIR.TAG`；没有重名条目或链接条目。三个大权重文件大小不同，未发现完全相同文件的重复副本。没有进行张量级去重或文件完整性哈希验证。两个 `best.ckpt` 是不同用途的模型，不能因同名删除。

Windows 下载目录：`C:\Users\wzh_z\spurs-download`。其中 `spurs-offline/` 与 `spurs-offline.tar` 是未打包数据和打包副本，只需上传 tar；`.venv/` 和 `download_spurs.py` 不需要上传。

当前源码初始化还会读取 ESM 权重及其配套权重；仅有 SPURS 检查点不足以完成默认初始化。当前分支的 ProteinMPNN 初始化没有独立下载权重步骤。参见 [ESM adapter](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/modules/esm2_adapter.py)、[ESM 官方加载器](https://github.com/facebookresearch/esm/blob/main/esm/pretrained.py)、[ProteinMPNN 初始化](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/org_transfer_model.py)。

## 4. 启动环境与验收命令

`/root/software/SPURS/offline-env.sh` 内容：

```bash
export HF_HUB_CACHE="$HOME/models/spurs-offline/hf_hub"
export TORCH_HOME="$HOME/models/spurs-offline/torch"
export HF_HUB_OFFLINE=1
```

变量应在 Python 导入 Hugging Face 包之前设置。即使使用名字带 `_from_hub` 的加载函数，也可以从缓存离线加载。`HF_HUB_OFFLINE=1` 避免加载前的联网 HEAD 请求。[Hugging Face 环境变量说明](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

标准启动方式：

```bash
cd /root/software/SPURS
source ./offline-env.sh
set -o pipefail
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python -u verify_spurs.py 2>&1 | tee verify_spurs.log
```

期望的完整验收标志：

```text
Single prediction: PASS
Multi prediction: PASS
SPURS installation verified.
```

已提供的 `verify_spurs.py` 使用仓库示例 `data/inference_example/DOCK1_MOUSE.pdb`、链 A；检查单点矩阵形状、有限值、野生型列近零，以及两组双点突变的输出。脚本先运行单点，再释放引用与 GPU 缓存，运行多点。它不是 1CXI 生产脚本。

`.bashrc` 自动加载只针对相应 shell 会话。systemd、cron、其他用户和非交互任务应显式设置环境；不要依赖它们读取 `.bashrc`。全局设置 `HF_HUB_OFFLINE=1` 会影响同用户其他 Hugging Face 程序，需要联网时可在该终端 `unset HF_HUB_OFFLINE`。

## 5. 已遇到的问题与处理

### 5.1 缺少 lmdb / atom3d

最后一次实际失败：

```text
spurs/datamodules/datasets/utils.py
ModuleNotFoundError: No module named 'lmdb'
```

该模块顶部还导入 `atom3d.datasets.LMDBDataset` 和 `tqdm`。它们没有被精简推理依赖清单完整覆盖。[解析模块源码](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/datamodules/datasets/utils.py)

已建议、尚待确认执行结果：

```bash
cd /root/software/SPURS
python -m pip install lmdb 'atom3d==0.2.6' tqdm -c constraints-server.txt
python -m pip check
python -c 'from spurs.datamodules.datasets.utils import alt_parse_PDB, get_pdb; print("Parser import: PASS")'
```

`constraints-server.txt` 由安装脚本记录当前 torch 版本，并约束 `numpy==1.26.4`。预期 torch 为 `2.8.0+cu128`；先读文件核实。补装可能引入其他传递依赖，不能声称上述命令已经解决全部 Python 3.12 兼容问题。

不要执行旧 `requirements.txt`，避免拉入 PyTorch 1.12 等训练依赖。当前安装使用 `requirements.inference.txt` 和 `pip install -e .`。[推理依赖](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/requirements.inference.txt)

### 5.2 配置没有 name 字段

Windows 下载脚本曾因 `config["model"]["name"]` 抛出 KeyError。配置允许省略这个字段；官方单点、多点默认值都是 `esm2_t33_650M_UR50D`。读取原始配置时使用：

```python
name = config["model"].get("name", "esm2_t33_650M_UR50D")
```

该修正也适用于离线缓存检查脚本，无需为此修改模型权重或官方配置。[单点默认配置](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/spurs.py)、[多点默认配置](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/spurs_multi.py)

### 5.3 torch.load 警告

PyTorch 2.6 起默认权重加载行为发生变化。当前 SPURS 使用旧检查点，启动命令临时设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。由此出现的 `forcing weights_only=False` 是预期提示，不是此次缺包错误。

仅对官方可信权重使用此兼容设置，不要把它当作对未知检查点的通用解决办法。[PyTorch 序列化说明](https://docs.pytorch.org/docs/2.9/notes/serialization.html)

### 5.4 其他警告

`mlp.py` 中 `if ckpt_path is '':` 会产生 SyntaxWarning；本次中断原因是缺少 lmdb。不要为消除提示而顺便重构模型代码。

## 6. 推理接口与开发约定

以下是接口摘要；以服务器实际安装的源码为最终依据。官方接口参考：[spurs/inference.py](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/inference.py)。

| 接口/调用 | 用法 |
|---|---|
| `get_SPURS_from_hub(device="cuda")` | 返回单点模型和 cfg，可读取离线缓存 |
| `get_SPURS_multi_from_hub(device="cuda")` | 返回多点模型和 cfg，可读取离线缓存 |
| `get_SPURS(ckpt_path, device="cuda")` | 单点本地模型目录；目录含配置和检查点，仍需 ESM 缓存 |
| `parse_pdb(path, name, chain, cfg, device="cuda")` | 生成模型输入字典 |
| `parse_pdb_for_mutation(groups)` | 返回位置张量与 WT/突变氨基酸编码张量 |
| `model(batch, return_logist=True)` | 单点模型的全替换矩阵；注意参数实际拼写为 `logist` |
| `model(batch)` | 多点模型对指定组合评分 |

单点输出按解析序列排列，形状为 `[L, 20]`。氨基酸列顺序为 `ACDEFGHIKLMNPQRSTVWY`，野生型对应列经归一化应为零。[单点实现](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/spurs.py)

多点输入形式为 `[["V2C", "P3T"], ["W1A", "V2Y"]]`。解析得到的两个张量分别赋给 `batch["mut_ids"]`、`batch["append_tensors"]`，移到匹配设备。输出按输入组合顺序对应；处理单个组合时用 `reshape(-1)` 避免标量形状问题。

开发要求：

- 显式 `model.eval()`，推理放在 `torch.no_grad()` 内；默认使用 float32，暂不引入混合精度或 compile。
- 模型在进程内加载一次，不要为每个突变重载几 GB 权重。
- 同一批组合先按突变数分组：当前 mutation parser 直接创建规则张量，不能直接把不同长度列表混在一起。
- 输入校验：WT 字母匹配解析序列、位置有效、无同位点冲突、仅接受当前模型支持的氨基酸。
- 多点得分使用多点模型，不用单点分数求和替代。
- forward 会往输入字典写入中间结果；多点实现还会 reshape/替换部分输入。重复调用时恢复突变输入，避免沿用已改变结构的数据。
- 对 1CXI 的长度和组合批量显存尚无实测；批量从小开始，记录显存，不预设整个搜索能一次放入 5090。
- 在确认中间张量确实与突变组合无关之前，不自行缓存内部表示来替代模型 forward。
- 不把非有限值、失败或缺失结果替换为有利分数；单独记录错误状态。
- 将以上内容视为开发约定，不视为已实现功能。

多点张量行为依据 [SPURSMulti 实现](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/spurs_multi.py)。

## 7. 1CXI 编号映射：生产脚本的首要检查

**原始 PDB 残基编号、成熟蛋白/文献编号和模型解析序列位置不能默认相等。** 突变字符串中的数字由当前接口减一后作为序列索引；不会自动转换 PDB 编号。

必须生成显式映射表，至少含：

```text
chain, pdb_resseq, insertion_code, wt_aa, model_position_1based
```

将映射与模型真实使用的 `batch["seq"]` 逐项核对。不能简单假定 BioPython 枚举结果与模型解析结果相同；缺失残基、非标准残基、插入码和序列过滤均可能改变对应关系。官方 `parse_pdb` 还尝试将 `resn_list` 转为整数，插入码需要专门核实处理。

用户的阳性对照为 **N188D/K192R**，尚未确认其编号体系。先确认 A 链对应编号确实为 N 和 K，再转换成模型位置。遇到不匹配应停止该条预测并报告，不能静默改 WT 字母或挪动位置。

建议输出同时保留：用户/原始编号突变字符串、模型编号突变字符串、映射文件、PDB 文件哈希。

## 8. 项目筛选规则与未决项

本节来自用户本地 `Thermal Stability Optimization.md`，属于项目方案，**不是 SPURS 官方性能保证或经过本次验证的阈值**。正式实现前重读该文件，以用户最新版本为准。

### 单点与其他过滤

- 保护/排除位点：ConSurf grade >= 8、配体重原子距离 < 6 Å、Ca²⁺ 距离 < 5 Å、二硫键 Cys。
- SPURS 单点候选：项目采用预测值 < -0.5，按升序筛选。
- Rosetta：项目采用 ΔΔG < -2.0 kcal/mol。
- FoldX：运行 5 次取平均，项目采用平均 ΔΔG < -1.0 kcal/mol。
- 基于耐热/普通同源组频率的排序在项目文档中注明“数据不够，不做了”。

后续 AI 必须核实 SPURS 输出的符号和单位与项目约定一致；不要凭变量名把所有模型输出直接标为 kcal/mol，也不要擅自翻转符号。论文入口：[Generalizable and scalable protein stability prediction with rewired protein generative models](https://www.nature.com/articles/s41467-025-67609-4)。本交接没有重新验证论文阈值解释。

### 双点兼容图

令 epistasis = pair_ddg - single_ddg_i - single_ddg_j。

- 严重不兼容：pair_ddg > -0.5，或 epistasis > 1.0，或属于同一位点。
- 高可信兼容：pair_ddg < -1.2 且 epistasis < 0.5。
- 其余为中间类别。

这是用户计划的分类规则。单点与多点分别来自两个预测器，跨模型差值作为项目指标，不应宣称是直接测得的物理相互作用能。

### Beam search

- 从双点开始，到最多 8 点；beam width = 100。
- 优先按多点模型总分升序。
- 扩展时与已有全部突变均不得严重不兼容，且满足边际贡献阈值。
- 配额：80% 高可信，20% 中间；高可信不足时由中间补齐。
- 高可信边比例定义为高可信边数 / C(t, 2)；配置项 `high_edge_ratio=0.80`。
- 单突变出现频率限制为 0.60B。
- 同层距离 d(S1,S2) = |S1| - |S1∩S2|。

**原文存在需要先确认的冲突，不能静默任选：**

1. 文字多样性阈值：2–4 点为 1、5–6 点为 2、7–8 点为 3；但数组为 `[0,1,1,1,2,2,2,3]`，若对应 1–8 点，则 7 点不同。
2. 文字边际阈值：3–4 点 -0.5、5–7 点 -0.3、8 点 -0.2；但数组为 `[0,0,-0.5,-0.5,-0.3,-0.3,-0.2,-0.2]`，若对应 1–8 点，则 7 点不同。
3. 边界相等如何归类、频率限制的 B 是否用固定 beam width 或实际层大小、候选不足时如何放宽其他条件，尚未明确。

## 9. 建议交付的脚本能力

下列为后续实现建议，不代表已经存在：

1. 环境/离线缓存诊断和官方示例 smoke test。
2. 1CXI A 链结构清洗、编号映射及保护位点表。
3. 单点全扫描，输出完整分数和按项目规则过滤的候选。
4. 对显式组合列表批量调用多点模型，支持断点续跑。
5. 双点兼容图和参数化 beam search，先解决第 8 节冲突。
6. 每次运行保存配置、torch/Python/SPURS 版本、模型 revision、输入哈希、日志、失败记录。

推荐表格字段：`mutation_original`、`mutation_model`、`mutation_count`、`spurs_score`、`model_kind`、`status`、`error`；单点另含原始残基编号、模型位置、WT/MT 字母和保护位点原因。未验证单位前用 `spurs_score`，不要在字段名中硬编码单位。

生产脚本接受 CLI 参数，不硬编码仅适用于示例 DOCK1_MOUSE 的突变。使用项目独立输出目录；不覆盖原始 PDB、ConSurf 文件或模型缓存。

## 10. 续接时的最小核查

在服务器当前环境执行：

```bash
cd /root/software/SPURS
source ./offline-env.sh
python -V
python -m pip show torch numpy spurs lmdb atom3d
git branch --show-current
git rev-parse HEAD
cat constraints-server.txt
python -m pip check
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u verify_spurs.py
```

这些是服务器 Bash 命令。不要把本地 Windows 的工具配置或依赖管理方式当作远端服务器环境的一部分。

本交接文档保存于本地项目目录；服务器部署状态仅依据用户提供的日志，尚未通过远程 SSH 独立检查。
