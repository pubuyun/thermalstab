# SPURS 单点筛选与组合搜索

只执行 SPURS 单点筛选、完整双点兼容图和 Diverse Beam Search。可突变位置使用第一阶段结果；已移除阳性对照，不运行 Other Evaluation Methods。

## 启动和输出目录

在已安装 SPURS 的服务器 Python 环境中，显式指定 YAML：

```bash
bash run_spurs.sh --config spurs_config.yaml --check
bash run_spurs.sh --config spurs_config.yaml --smoke-test
bash run_spurs.sh --config spurs_config.yaml --stage all
```

`--config` 必填，没有隐式默认配置。`--check` 检查离线文件、依赖导入、CUDA 运算和真实 SPURS 编号映射。`--smoke-test` 从输入结构自动构造两组 API 探针，检查前向、批量和重复调用一致性；不假定稳定化效果，不写入候选或搜索缓存。

结果直接保存到 **YAML 文件旁的同名目录**：

| 配置文件 | 结果目录 |
|---|---|
| `spurs_config.yaml` | `spurs_config/` |
| `configs/run_a.yaml` | `configs/run_a/` |
| `/data/experiments/depth10.yml` | `/data/experiments/depth10/` |

只去掉最后一个 `.yaml` / `.yml` 扩展名，例如 `run.v2.yaml` 对应 `run.v2/`。没有 `output_dir` 配置或输出路径覆盖参数。新实验复制并重命名 YAML 即可；相同文件名再次运行会检查输入、配置、代码和环境身份，再续跑。

默认输入是**项目根目录**的 `1CXI.pdb`、`mutable_positions.txt` 和链 A，与当前工作目录或 YAML 所在位置无关。需要改变时使用命令行：

```bash
python scripts/run_spurs.py --config configs/run_a.yaml \
  --pdb /data/protein.pdb --chain A \
  --mutable-positions /data/mutable_positions.txt
```

显式指定的相对输入路径相对于当前工作目录。Python 和 Bash 入口都可从其他工作目录运行，不启动服务、不构建项目、不重新安装 SPURS。

## YAML 格式与 depth

`spurs_config.yaml` 只含方法文档中的四个部分：

```yaml
single:
  cutoff: -0.5

pair_graph:
  severe:
    pair_ddg_min: -0.5
    epistasis_max: 1.0
  high_confidence:
    pair_ddg_max: -1.2
    epistasis_max: 0.5

beam:
  depth: 8
  width: 100
  high_confidence_quota: 0.8
  high_edge_ratio: 0.80
  frequency_limit: 0.60

gradient:
  diversity: [0, 1, 1, 1, 2, 2, 2, 3]
  marginal: [0, 0, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2]
```

没有 `source`、`positive_controls`、`runtime`、`output_dir`、`input` 或 `frequency_denominator` 配置项。额外键和重复 YAML 键会报错，避免拼写错误静默生效。

`beam.depth` 可以是任何 **≥ 2 的整数**，不再限制为 8。两个 gradient 数组按 `突变数 - 1` 索引，长度必须至少覆盖目标 depth；多余元素不参与搜索。depth=4 可使用原八项数组，也可只保留前四项。depth>8 时必须显式添加后续每层的距离和边际阈值；程序不会复制最后一项或猜测新阈值。数组不足会在加载模型前报错。

默认数组仍沿用已确认的文档末尾版本：7 点最小距离 2、边际上限 -0.2。双点由兼容图初始化，不检查边际阈值。

## 筛选和搜索规则

| 步骤 | 规则 |
|---|---|
| 单点 | 全链 `[L,20]` 矩阵；仅保留 mutable 位置、非 WT、分数严格 `< single.cutoff` 的全部突变，无 top-K 截断 |
| 双点 | 所有候选的无序对；不同位置调用多点模型，同位置直接严重不兼容 |
| epistasis | 多点模型双点分数减去两个单点模型分数 |
| 严重边 | 双点分数 `>` severe 阈值，或 epistasis `>` severe 阈值，或同位置；优先判定 |
| 高可信边 | 双点分数 `<` high_confidence 阈值，且 epistasis `<` 对应阈值 |
| 中间边 | 剩余双点，遵守上述严格不等式 |
| 扩展 | 新突变与已有全部突变都不严重不兼容；`child_score-parent_score <= marginal[t-1]` |
| 多路径 | 子组合只需有一条来自当前 beam 的路径达到边际；保留边际最优的父组合，子组合只评分一次 |
| 高可信池 | 组合全部边中高可信边占比 `q >= high_edge_ratio`，分母 `C(t,2)` |
| 配额 | 各池按多点总分升序贪心选择，先最多 `floor(B × high_confidence_quota)` 个高可信，再由中间池补至 B |
| 多样性 | 对每个已选组合同层距离 `t-共同突变数 >= diversity[t-1]` |
| 频率 | 任一突变最多出现 `floor(B × frequency_limit)` 次，B 固定为配置的 width |
| 候选不足 | 保持约束，允许 beam 不满；高可信不足由中间补齐，中间不足不反向增加高可信配额 |

组合分数始终来自多点模型的完整前向，不用单点求和替代。默认 float32、eval、no_grad，每次深复制原始 batch 并重新设置突变张量。组合批量显存不足会递减批量，单组合仍失败则记录错误并停止。单点模型释放后才加载多点模型。

## 环境和依赖

YAML 只描述方法参数，默认沿用交接环境。更换机器时可使用环境变量：

| 环境变量 | 默认值 |
|---|---|
| `SPURS_REPO` | `/root/software/SPURS` |
| `SPURS_OFFLINE_ROOT` | `/root/models/spurs-offline` |
| `SPURS_MODEL_REVISION` | `0cc7a565af8f31eb122819f95a9d16e27b3d1596` |
| `SPURS_DEVICE` | `cuda` |
| `SPURS_BATCH_SIZE` | `8` |
| `SPURS_SEED` | `42` |

离线缓存及可信旧版权重兼容变量会在导入 SPURS/Hugging Face 前设置，不依赖 `.bashrc`。实际输入、运行环境、版本与模型身份仍记录在 `manifest.json`，可复用的方法快照为 `config.resolved.yaml`。

YAML 使用 PyYAML（通常已随 SPURS 的 OmegaConf 安装）。绘图仅需 numpy 和 matplotlib，可在没有 GPU / SPURS 的电脑运行。按需安装：

```bash
python -m pip install -r requirements-tools.txt
```

服务器的SPURS环境必须追加已有约束，避免工具依赖升级NumPy：

```bash
python -m pip install -r requirements-tools.txt \
  -c /root/software/SPURS/constraints-server.txt
```

该约束记录已成功运行的`numpy==1.26.4`及当前torch版本；不要执行SPURS旧训练依赖，也不要重装torch。

如果预检报`expected np.ndarray (got numpy.ndarray)`，先在同一Python中运行：

```bash
python - <<'PY'
import sys, numpy, torch
print(sys.executable, numpy.__version__, numpy.__file__, torch.__version__)
print(torch.from_numpy(numpy.zeros(1, dtype=numpy.float32)))
PY
```

若最小探针也失败，先检查是否混用了pip与conda的NumPy：

```bash
python -m pip show numpy
/root/miniconda3/bin/conda list | grep -E '^numpy([[:space:]]|-)'
```

当前验证环境是`/root/miniconda3/bin/python`。如果输出同时含pip安装的NumPy与conda的`numpy-base`，仅用pip覆盖可能留下混合文件。先移除pip记录，再让conda统一恢复两个包：

```bash
/root/miniconda3/bin/python -m pip uninstall -y numpy
/root/miniconda3/bin/conda install -y --freeze-installed --force-reinstall \
  'numpy=1.26.4' 'numpy-base=1.26.4'
```

关闭当前shell中可能驻留的Python/Jupyter进程，启动新进程后再次执行转换探针。确认探针成功后再运行SPURS。不要同时执行pip和conda的NumPy安装，也不要重装torch。

如果`conda list`本来就只有一致的1.26.4包，先保存以下输出再处理；它用于区分Torch安装损坏与SPURS导入污染：

```bash
/root/miniconda3/bin/python - <<'PY'
import numpy, torch
print('before SPURS:', torch.from_numpy(numpy.zeros(1, dtype=numpy.float32)))
import sys
sys.path.insert(0, '/root/software/SPURS')
import spurs.inference
print('after SPURS:', torch.from_numpy(numpy.zeros(1, dtype=numpy.float32)))
PY
```

## 单独绘图

指定**结果文件夹**即可，无需再传配置；脚本读取该目录保存的方法快照，自动识别可用中间结果：

```bash
# 所有已完成阶段
python scripts/plot_spurs.py spurs_config

# 分别运行
python scripts/plot_spurs_single.py spurs_config
python scripts/plot_spurs_pairs.py spurs_config
python scripts/plot_spurs_beam.py spurs_config

# 选择类型与导出格式
python scripts/plot_spurs.py configs/run_a --plots pairs beam \
  --formats png pdf --sample-size 20000 --matrix-size 40 --top-mutations 30
```

输出在结果目录下的 `plots/`，默认同时生成 PNG 和 PDF，另支持 SVG。使用无窗口后端，适合远程服务器。旧版 JSON 配置生成的结果目录也能直接绘制。

| 图文件名 | 内容 |
|---|---|
| `single_scores` | 可变位置非 WT 单点评分分布、已选分布、真实 cutoff；各位置最优分数 |
| `single_substitution_heatmap` | 替换氨基酸 × PDB 位置热图；灰色表示保护位点、WT 或缺失 |
| `pair_scores_and_classes` | 全部有效双点的三类数量；双点总分与 epistasis 散点及保存的阈值线 |
| `pair_compatibility_matrix` | 默认单点分数最好的前 40 个突变之间的三类兼容矩阵，明确标注是子集 |
| `beam_progress` | 自动识别所有深度：每层总分分布、最优分数、beam 占用、高可信边比例、边际分布 |
| `beam_mutation_frequency` | 常见突变占实际已选组合的比例；每层最大出现次数与固定 B 的频率上限 |

大双点表逐行读取。类别计数使用全部有效边；散点最多均匀抽样 `sample-size` 条（固定种子 42），图题和报告注明抽样量，同位点未评分冲突不会变成零分散点。兼容矩阵只展示限定数量节点，不截断预测数据。无保存配置时仍可绘制数据，但不猜测阈值线。

单点完成即可绘制单点图；其他阶段尚无输出时自动跳过。beam 按当前已有层文件绘制，允许空层、提前耗尽或 depth>8。只读取已完成的 CSV，不读 `.tmp`。`plot_report.json` 和每类 `*_plot_report.json` 记录生成文件、数据量、抽样量、跳过/失败状态和无效行数。重复绘图不运行模型，也不修改 CSV。

## 分阶段、结果与恢复

```bash
bash run_spurs.sh --config spurs_config.yaml --stage single
bash run_spurs.sh --config spurs_config.yaml --stage pairs
bash run_spurs.sh --config spurs_config.yaml --stage search
```

后续阶段自动补齐前序计算，成功预测会复用。也可用 `--stage score --mutations combinations.txt` 评价显式组合：每行用 `/` 分隔 PDB 编号突变，至少两处、最多不超过结构残基数；不同突变数分组调用，写入 `specified_combinations.csv`，不注入搜索。

主要输出：`residue_mapping.csv`、`single_all.csv`、`single_candidates.csv`、`pair_graph.csv`、`beam_depth_<n>.csv`、`beam_all_depths.csv`、`final_candidates.csv`、`search_summary.json`、`predictions.sqlite3`、`failed_predictions.csv`、`manifest.json`、`config.resolved.yaml`、`run.log`、`run_status.json`。

`final_candidates.csv` 只放目标深度结果，提前耗尽则为空，较低层仍在各层 CSV 和汇总中。所有双点都构图，计算量随单点候选数量平方增长。

中断后重跑相同命令，SQLite 已提交的批次不会重算，失败批次重试。CSV 完成后原子替换，进程锁退出时自动释放。配置、输入、代码或环境变化会拒绝复用旧缓存；新实验改用新的 YAML 文件名。大权重以固定 revision、大小和 mtime 识别，模型 YAML 额外 SHA256，不等于大权重全文完整性校验。

## 验证

```bash
python scripts/run_spurs.py --config spurs_config.yaml --validate-inputs
python -m unittest discover -s tests -v
```

输入检查需要 PyYAML，不需要 torch。真实模型映射和 GPU 推理由服务器上的 `--check` / `--smoke-test` 验收。测试中的合成分数仅用于验证算法和图表，不是实际预测。

SPURS 原始输出不翻转符号，阈值和跨两个模型计算的 epistasis 属于本项目约定。[论文](https://www.nature.com/articles/s41467-025-67609-4)、[推理接口](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/inference.py)、[多点前向](https://github.com/luo-group/SPURS/blob/fix/py311-inference-beta/spurs/models/stability/spurs_multi.py)。
