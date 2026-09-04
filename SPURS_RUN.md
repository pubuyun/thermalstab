# SPURS 后两阶段运行说明

本实现只使用 SPURS：阶段 2 为单点生成与筛选，阶段 3 为双点兼容图与最多 8 点的 Diverse Beam Search。不运行方法文档中的 Other Evaluation Methods，不重新计算第一阶段的可突变位点。

## 在远程服务器运行

将整个项目同步或 Git 拉取到服务器，保留 `1CXI.pdb`、根目录的 `mutable_positions.txt`、`spurs_config.json`、`scripts/` 和 `run_spurs.sh`。进入项目根目录，使用已经装好 SPURS 的 Python 环境：

```bash
python -V
bash run_spurs.sh --check
bash run_spurs.sh --smoke-test
bash run_spurs.sh --stage all
```

`--check` 检查离线文件、Python 包导入、CUDA 运算、两个模型各自配置下的真实 SPURS 解析结果，将诊断映射写入 `preflight_mapping.csv`。`--smoke-test` 在 **1CXI A 链**执行完整单点前向和多点前向，检查单组合输出、批量输出和重复前向一致性；成功日志包含 `Single prediction: PASS` 和 `Multi prediction: PASS`。阳性对照的预测不要求人为通过稳定化阈值。

`--stage all` 也会自动预检，完成单点、对照、完整双点图和组合搜索。以上命令不会安装依赖、启动其他服务或联网下载权重。不要求项目放在 SPURS 源码目录，不要求存在特定名称的 Conda 环境，也不依赖 `.bashrc`。

也可从任意目录直接运行：

```bash
python -u /path/to/thermalstab/scripts/run_spurs.py --config /path/to/thermalstab/spurs_config.json --stage all
```

已提供 LF 换行规则，使用 `bash run_spurs.sh` 无需设置可执行位。

## 唯一配置文件

`spurs_config.json` 包含输入、输出、远程路径、模型 revision、批量大小、随机种子、所有筛选与搜索阈值、阳性对照。

默认使用交接文档里的 `/root/software/SPURS`、`/root/models/spurs-offline` 和模型 revision `0cc7a565af8f31eb122819f95a9d16e27b3d1596`。路径不同只需修改 `runtime.spurs_repo` / `runtime.offline_root`。输入和输出的相对路径均相对于配置文件所在目录。

程序会在导入 Hugging Face、torch、SPURS 之前设置 `HF_HUB_CACHE`、`TORCH_HOME`、`HF_HUB_OFFLINE=1`、`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，对接交接说明中的可信旧版权重。`refs/main` 必须等于配置中的 revision，避免静默加载其他版本。

默认设备为 `cuda`，组合批量为 8；CUDA 不可用时报错，不会自动切到 CPU。组合前向显存不足会递减批量，单组合仍不足则保存错误并终止。单点扫描本身是整个 `[L,20]` 矩阵的一次前向，不受组合批量控制。运行日志记录峰值 GPU 显存。默认 float32、eval、no_grad，不启用混合精度或 compile，也不自行缓存模型内部表示。

## 方法与明确约定

| 项目 | 实际规则 |
|---|---|
| 单点 | 所有可突变位置的 19 种非 WT 替换；保留 SPURS 分数严格 `< -0.5` 的全部突变，无 top-K 截断 |
| 双点 | 为所有候选的无序对建图；不同位置用 SPURS **多点模型**预测，同位置直接标为严重不兼容 |
| epistasis | `pair_score - single_i - single_j`；单点分数来自单点模型 |
| 严重不兼容 | `pair_score > -0.5` 或 `epistasis > 1.0` 或同一位置，优先判定 |
| 高可信边 | `pair_score < -1.2` 且 `epistasis < 0.5` |
| 中间边 | 剩余双点；等于边界时按上述严格不等式分类 |
| 组合高可信池 | 高可信边比例 `q >= 0.80`；分母是 `C(t,2)`，包括组合内全部边 |
| 扩展 | 对当前层每个 beam 组合尝试添加每个候选突变，新突变与已有全部突变都不能严重不兼容 |
| 边际 | `multi_score(child) - multi_score(parent) <= marginal[depth-1]`；等于阈值允许 |
| 多路径 | 同一子组合可由多个父组合产生，只需有一个当前 beam 父组合满足边际；保留边际最优的父组合，子组合只评分一次 |
| 排序 | 各池按多点总分升序，平分时按规范化突变字符串排序，最后输出也按总分升序 |
| 配额 | 先选最多 `floor(0.8B)` 个高可信，再由中间池补至 B，包含补足高可信池的缺额 |
| 距离 | 同层 `d(S1,S2)=t-|S1∩S2|`，按完整突变身份比较；与每个已选组合都必须达到阈值 |
| 频率 | 每个单点突变最多出现 `floor(0.60B)` 次；**B 明确采用配置的 beam width（默认 100）** |
| 候选不足 | 不放宽兼容性、边际、距离或频率；中间池不足时也不反向增加高可信配额，允许 beam 不满 |

配额与多样性共同通过“池内分数升序、逐个检查接纳”的确定性贪心选取实现。受约束后可以少于 B，不承诺全局最优的 beam 子集。

按照用户确认，采用原方法文档**末尾数组**，数组索引对应 1–8 点，实际从双点开始：

| 突变数 | 最小距离 | 加入新突变的边际上限 |
|---|---:|---:|
| 2 | 1 | 不使用，双点直接由兼容图初始化 |
| 3 | 1 | -0.5 |
| 4 | 1 | -0.5 |
| 5 | 2 | -0.3 |
| 6 | 2 | -0.3 |
| 7 | 2 | -0.2 |
| 8 | 3 | -0.2 |

不使用正文中与数组冲突的 7 点距离 3、边际 -0.3。

## 编号和阳性对照

脚本独立读取 PDB 的残基身份和完整 N/CA/C/O 骨架，再与官方 `alt_parse_PDB` 的 `resn_list`、序列和实际 `batch['seq']` 逐项核对。`residue_mapping.csv` 同时记录链、PDB 编号、插入码、WT、模型 1-based 位置。

本项目本地输入有 686 个 A 链残基、453 个可突变位点，单点候选空间为 8607 个替换。N188 和 K192 分别是 ASN、LYS。当前结构没有内部编号缺口、插入码或缺失骨架原子。脚本支持起始编号不为 1 的连续 PDB，并显式转换模型位置；对当前上游接口不能安全处理的内部缺号、插入码、非标准氨基酸、多模型和骨架 altloc 明确报错，不静默删除或重编号。

`N188D/K192R` 单独评分并写入 `positive_controls.csv`。K192 不在提供的可突变列表内，作为对照仍可评分，但不会因此进入搜索候选。对照 WT 不匹配会终止，不修改 WT 或移动位置。

## 分阶段与指定组合

```bash
bash run_spurs.sh --stage single
bash run_spurs.sh --stage pairs
bash run_spurs.sh --stage search
```

后续阶段自动补齐必需的前序计算，已有单点矩阵和多点评分会复用；`search` 重建双点类别/候选池时复用 SQLite 评分，不重复已完成的模型推理。

对已知组合额外评分时，新建一个文本文件，每行用 `/` 分隔 PDB 编号突变，例如：

```text
N188D/K192R
```

```bash
bash run_spurs.sh --stage score --mutations combinations.txt
```

每组允许 2–8 个突变；不同突变数分组调用多点模型。输出 `specified_combinations.csv`，不改变单点候选或搜索规则。该模式允许像对照一样评价非 mutable 位置。

## 结果和续跑

默认目录为 `results/spurs/`（已加入 Git 忽略）：

| 文件 | 内容 |
|---|---|
| `residue_mapping.csv` | 与 SPURS 解析输入核对后的编号表 |
| `single_all.csv` / `single_matrix.json` | 全链所有 20 列原始预测，含 WT 零列与 mutable 标记 |
| `single_candidates.csv` | 所有通过单点阈值的可变位置非 WT 替换，升序 |
| `positive_controls.csv` | 阳性对照多点预测 |
| `pair_graph.csv` | 全部无序双点的分数、epistasis 和三类边，包括同位点严重冲突 |
| `beam_depth_2.csv` … `beam_depth_8.csv` | 各层已选组合、原始/模型编号、总分、q、父组合、边际 |
| `beam_all_depths.csv` | 合并各层 beam；排名仍是层内排名 |
| `final_candidates.csv` | 仅目标深度的最终组合；提前耗尽则为空，不将低阶结果伪装为 8 点结果 |
| `search_summary.json` | 完成/耗尽状态、各层配额、频率、候选量及最后非空层 |
| `predictions.sqlite3` | 已评分组合的缓存和当前搜索池；每个完成批次事务提交 |
| `failed_predictions.csv` | 当前未成功的多点组合、错误状态及消息 |
| `manifest.json` / `config.resolved.json` | 输入 SHA256、配置快照、Python/依赖版本、SPURS commit/源码哈希、模型身份 |
| `run.log` / `run_status.json` | 日志、峰值显存、运行状态与失败原因 |

中断后运行**相同命令**即可恢复。成功批次不会重算；之前失败的批次会重试；双点图与层选择从缓存重新生成以保证完整性。进程锁会随进程退出自动释放，避免两个写入任务破坏同一输出目录。尚在写入的 CSV 使用临时文件，完成后原子替换。

配置、输入、代码、模型或依赖环境改变后，同一输出目录会拒绝混用旧结果。要改变阈值或路径开展新实验，请设置新的 `output_dir`。模型身份使用固定 revision、文件大小与纳秒 mtime，模型 YAML 额外计算 SHA256；几个 GB 的权重不做逐次全文哈希，这不是权重内容完整性验证。SQLite 与已写出的结果也不应手工修改。

全双点预测数量随单点候选数 N 按 `N(N-1)/2` 增长，实际耗时依赖筛出的 N 和远程 GPU。程序不会为加速而截断候选。图类别使用三角形字节数组，搜索候选和分数存入 SQLite，避免将所有双点预测对象堆入内存。

## 验证边界

本地不需要 torch 的检查：

```bash
python scripts/run_spurs.py --validate-inputs
python -m unittest discover -s tests -v
```

`--validate-inputs` 只核对配置、PDB、mutable 和对照，输出名称明确为 `input_mapping_unverified.csv`，不会声称模型映射或 GPU 推理已验收。测试使用明确标记的合成预测，覆盖完整候选图、三类边阈值、非加和评分、7 点数组、边际、多路径去重、配额、多样性、频率限制、8 点搜索、失败重试和断点续跑。生产 CLI 没有假模型/模拟分数开关。

当前开发环境未连接到目标服务器执行真实推理，因此远程验收以服务器上 `--check`、`--smoke-test` 的成功日志为准。缺包时沿用交接说明补齐现有环境，不重新安装旧训练版依赖。

## 上游依据

- 评分使用上游接口的原始输出，不翻转符号；论文以负值作为稳定化方向，结果字段保持 `spurs_score`，不混同实验测量值。[SPURS 论文](https://www.nature.com/articles/s41467-025-67609-4)
- 单点、多点加载及突变位置减一等接口依据：[兼容分支 inference.py](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/inference.py)。
- 多点使用独立模型的完整前向；每次深复制原始 batch 并重新设置 mutation tensors，以应对上游输入修改。[SPURSMulti](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/spurs_multi.py)

上述兼容边、跨两个模型计算的 epistasis 指标和 beam 阈值属于本项目的方法约定，不是实测物理相互作用能，也不保证实验热稳定性结果。
