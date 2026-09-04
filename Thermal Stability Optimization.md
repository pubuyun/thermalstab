# Thermal Stability Optimization

## 过滤

Pdb: 1CXI

MSA保守残基, 配体重原子距离\<6A, Ca²⁺\<5A, 二硫键Cys

MSA:

Rate4Site: https://academic\.oup\.com/bioinformatics/article/18/suppl\_1/S71/231818

By consurf: https://pmc\.ncbi\.nlm\.nih\.gov/articles/PMC4987940/

http://consurf\.tau\.ac\.il/

ConSurfGrade \>= 8 过滤

\[1CXI\_A\_With\_Conservation\_Scores\.pdb\]

\[1CXI\_A\_consurf\_grades\.txt\]

## 生成

https://www\.nature\.com/articles/s41467\-025\-67609\-4

论文阈值\<\-0\.5有稳定化效果

选\<\-0\.5的所有突变

## Other Evaluation Methods

### Evolution\-based 

MSA耐热同源CGTase残基

同源CGTAse分为耐热组和普通组

AH=候选残基在耐热组中的频率

WH=野生型残基在耐热组中的频率 反映耐热同源蛋白是否仍然倾向保留目标酶的野生型残基

AN=候选残基在普通组中的频率 判断候选残基是耐热β\-CGTase偏好的残基，还是所有β\-CGTase都普遍存在的残基

Q= max\(WH,AN\)/AH 越小越好 

条件: AH\>m% Q\<1/n等价于 

候选残基至少出现在m%的耐热β\-CGTase中

耐热组中，候选残基的频率是野生型残基的n倍

候选残基在耐热组中的频率是普通组的n倍

排序条件: **Q升**序

\[cgtase\_thermal\_stability\_sequences \(3\)\.csv\]

\[clustalo\-I20260903\-091622\-0822\-40368356\-p1m\.aln\-clustal\_num\]

相似性矩阵

\[clustalo\-I20260903\-091622\-0822\-40368356\-p1m\.pim\]

数据不够,不做了 

### Physics\-based 

**Rosetta ΔΔG \< −2\.0 kcal/mol**。

> FireProt: Rosetta −2 kcal/mol 和 FoldX −1 kcal/mol 时精确率较高、假阳性率较低
> 
> 

排序条件: **ΔΔG升**序

### Knowledge\-based

FoldX 5次取平均

**FoldX means ΔΔG \< −1\.0 kcal/mol** 

排序条件: **ΔΔG升**序







## 多点突变

### 兼容图

参考 \[fireprot\.pdf\]

构建所有双点突变\(i,j\)图, 三类边

1. 严重不兼容: ΔΔG\(i,j\) \> \-0\.5 或 ΔΔG\(i,j\) \- ΔΔG\(i\) \- ΔΔG\(j\) \> 1\.0 或属于同一位点

2. 中间值

3. 高可信兼容  ΔΔG\(i,j\) \< \-1\.2 \& ΔΔG\(i,j\) \- ΔΔG\(i\) \- ΔΔG\(j\) \< 0\.5

### Diverse Beam Search

#### 每轮扩展

从双点开始搜索。对于当前组合 S，候选突变 m 必须满足：

1. 与 S 中所有突变都不是严重不兼容； 

2. 加入后达到对应边际阈值。

选择顺序 SPURS总ΔΔG升序

高可信边比例: 对于t个突变组合S, q\(S\)=S中所有高可信边/C\(t,2\)

每层beam宽度80%来自高可信, 20%来自中间

如果高可信池不足80个，可以用中间池补齐

#### Diversity

1. 定义距离d\(S1,S2\)=S1\-相同突变数

每层最小距离: 

2\-4: 1

5\-6: 2

7\-8: 3

2. frequency\(m\)≤0\.6B, 任意单点突变最多出现在60%的组合中

### 边际 

每个新加入突变至少贡献

3\-4: \-0\.5

5\-7: \-0\.3

8: \-0\.2

直到8点突变

```Plain Text
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
    diversity: [0, 1, 1, 1, 2, 2 ,2, 3] 
    marginal: [0, 0, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2]
    
```

阳性对照组 N188D/K192R

