# OOAL 在 UMD 上的复现与可视化

## 1. 目的

得到 OOAL 在 UMD 测试帧上的逐帧预测，用来画定性对比图。

- 表 3 的 OOAL 一行仍引用原文（74.6 / 59.7 / 66.4），不用复现数字替换（例外见第 6 节）。
- 复现的预测无论数字高低都用于定性图；复现数字记进本地《实验细节_本地保留》，并写进图注，让读者知道图里的 OOAL 是什么水平。
- 用逐帧统计找出 OOAL 比本方法差的类别、可供性与错误类型，据此选图（第 7 节）。

## 2. 已知与未知

| 来源 | 给出了什么 | 没给什么 |
|---|---|---|
| 论文与补充材料 | seen = category split、unseen = novel split；7 种可供性；one-shot 54 / 76 张；“MMSegmentation 默认设置”；通用超参 SGD lr 0.01、20k 步、p=8、j=3、t=2、ViT-B | UMD 的输入分辨率、增强、batch、背景怎么处理、阈值 |
| 仓库 | 只有 AGD20K 的数据读取、训练、评测。`models/ooal.py` 与数据集无关；`utils/evaluation.py` 里还留着 `intersectionAndUnionGPU`（分割评测的残留） | UMD 的数据读取、训练与评测脚本 |
| issue | #1 UMD 下载地址；#3 作者反问过 mIoU 是否计入背景，答案没有公开；#12 有人（cjiang2）称已“按其他 issue”复现了 UMD 接入 | 代码本身 |

结论：模型可以原样用，需要补的只有数据、标签格式、读出和评测四处，其中评测直接复用你已有的 `umd_eval.py`。

## 3. 适配时做的决定

| 项 | 决定 | 理由 |
|---|---|---|
| 模型 | `models/ooal.py` 一行不改。新增 `OOALUMD.forward_all` 读出 7 个通道 | 原 `forward` 在测试时只返回被查询的那一个通道（AGD20K 每张图只问一种可供性）；UMD 要 7 个通道才能拼出标签图。评测时第一帧自动核对 `forward_all` 与原 `forward` 逐通道一致，差异超过 1e-5 就停止 |
| 类别与文本 | UMD 编号顺序 grasp, cut, scoop, contain, pound, support, wrap grasp | 与 `umd_eval.AFFORDANCE_NAMES` 一致，导入时检查 |
| 标签与损失 | 7 通道 one-hot，背景全 0，乘 255 后交给原来的 BCE | 原代码算 `BCELoss(pred, label / 255)`，这样不用改损失 |
| 增强 | 原 `TrainData.transform`：缩放 → 随机裁剪 → 水平翻转；尺寸从 256 → 224 改为 476×644 → 420×560 | UMD 原图 640×480、物体小（seen 放大倍数中位数 4.1），224 输入时物体只占几个 token。476×644 是最接近原图的 14 的倍数，与 MMSeg 默认尺度（短边约 512）相近；裁剪比例沿用 224/256 = 0.875 |
| 测试 | 整帧缩放到 476×644（训练尺度）；sigmoid 图双线性缩回原图；最大通道 > 0.5 取 argmax，否则背景 | 直接调用 `umd_eval.predict`，与本方法同一条规则；0.5 也是 BCE 的决策边界 |
| 优化 | 仓库默认：SGD 0.01、momentum 0.9、Nesterov、wd 5e-4、cosine、20k 步、batch 1、seed 321 | 论文没写 UMD 的 batch，不另设 |
| 选模型 | 评最后一步，不在测试集上挑 checkpoint；每 5000 步存一份 | 原 `train.py` 每 2000 步在测试集上按 KLD 选最好的一份，这里不这样做 |
| 评测 | 示例、测试帧、ladle 重映射、混淆矩阵、不含背景的 mIoU 全部从 `umd_eval.py` 导入 | 与本方法用同一份代码，口径不会走样 |

## 4. 环境

2026-10-11 服务器迁移补充：先按 [SERVER_HANDOFF.md](SERVER_HANDOFF.md) 配置数据路径、本地官方权重及隔离环境。已验证 torch 2.6.0 / torchvision 0.21.0；下方 torch 2.4.1 的安装示例仍是另一种匹配组合，不要只降级 torchvision。`models/ooal.py` 保持不改，`OOALUMD` 只在构造时接入本地加载器。

OOAL 的 README 写的是 Python 3.7 / PyTorch 1.12，但 `torch.hub` 拉到的 DINOv2 新代码要求 Python ≥ 3.10（issue #13 的报错就是这个）。下面仅适用于已有的 drawafford 环境确实使用 torch 2.4.1；其他环境按交接手册安装匹配组合：

```bash
conda create -n ooal --clone drawafford
conda activate ooal
pip install torchvision==0.19.1 ftfy regex tqdm matplotlib "setuptools<70"
```

- `torchvision==0.19.1` 对应环境里的 torch 2.4.1；
- `setuptools<70`：OOAL 自带的 CLIP 代码写的是 `from pkg_resources import packaging`，新版 setuptools 没有这个名字；
- 未配置本地权重时，第一次运行会联网下载 DINOv2（torch.hub）和 CLIP ViT-B/16（`~/.cache/clip`）；本地官方权重配置见交接手册第 5 节。

## 5. 运行

直接使用 fork `https://github.com/yhzhengoo/OOAL_umd`，无需再覆盖 zip。`DRAWAFFORD_ROOT` 同时支持 `affordance_research/umd_eval/umd_eval.py` 和 `umd_eval/umd_eval.py` 两种目录布局；`UMD_DATASET_ROOT`、`UMD_BASE_REF_ROOT`、`UMD_NOVEL_REF_ROOT` 可覆盖旧服务器路径。见交接手册的 `/home1` 示例。单 GPU 时两组都用 `--gpu 0` 串行运行。

```bash
git clone https://github.com/yhzhengoo/OOAL_umd && cd OOAL_umd
export DRAWAFFORD_ROOT=/实际位置/drawafford_aed
```

`DRAWAFFORD_ROOT` 必须指向目标机的代码根目录。按交接手册第 4 节配置 `UMD_DATASET_ROOT`、`UMD_BASE_REF_ROOT`、`UMD_NOVEL_REF_ROOT` 和 `OURS_ROOT`，再运行 `python -m ooal_umd.check_setup --ours_root "$OURS_ROOT"` 确认，不能假设旧服务器路径仍可用。

**第 0 步：CPU 自检（不下载权重，几秒）。**

```bash
python -m unittest ooal_umd.test_protocol_numpy ooal_umd.test_torch_cpu -v
```

`test_torch_cpu` 用小的随机模块代替 DINOv2 与 CLIP，检查 `forward_all` 与原 `forward` 一致、one-hot 标签下的 BCE、数据增强的尺寸与 ladle 重映射。

**第 1 步：冒烟（几分钟）。**

```bash
python -m ooal_umd.train_umd --split seen --iters 200 --save_step 200 --save_root runs/smoke
python -m ooal_umd.test_umd  --split seen --limit 50 \
    --model_file runs/smoke/seen/<时间>/iter_000200.pth --out_dir runs/smoke/eval
```

核对：`test_umd` 第一行打印的 `self-check ... max diff` 小于 1e-5；`runs/smoke/eval/pred/` 下有 50 张 png。

**第 2 步：正式训练，两个划分各一次。**

```bash
python -m ooal_umd.train_umd --split seen   --gpu 0
python -m ooal_umd.train_umd --split unseen --gpu 0
```

这里按单 GPU 串行运行；多 GPU 并行时才把 Unseen 的卡号改为实际存在的另一张卡。

**第 3 步：评测最后一步并汇总。**

```bash
S=runs/ooal_umd/seen/<时间>
U=runs/ooal_umd/unseen/<时间>
python -m ooal_umd.test_umd --split seen   --model_file $S/iter_020000.pth --out_dir $S/eval_020000
python -m ooal_umd.test_umd --split unseen --model_file $U/iter_020000.pth --out_dir $U/eval_020000 --gpu 0
python -m ooal_umd.report_umd --seen $S/eval_020000 --unseen $U/eval_020000 --out ooal_umd_report.md
```

核对：`summary.json` 里 `frames` 为 14020 / 7969；`report_umd` 输出与原文的差值和逐可供性 IoU。

**输出。**

| 文件 | 内容 |
|---|---|
| `eval_*/summary.json` | `umd_eval.scores` 的结果、混淆矩阵、checkpoint 的 sha256、自检差值 |
| `eval_*/per_frame.npz` | `ids`、`objects`、`cm`（N×8×8），与本方法的 `per_frame.npz` 同一格式 |
| `eval_*/pred/<id>.png` | 逐帧标签图，0 为背景，1–7 为 UMD 编号，画图用 |
| `ooal_umd_report.md` | seen / unseen / hIoU 与原文对照 |

## 6. 结果怎么用（看数字之前定好）

| 复现的 hIoU | 怎么用 |
|---|---|
| 高于本方法（66.6） | 定性图照常放 OOAL；同时把复现数字作为一行加进表 3（“OOAL，我们的复现，同一评测代码”），正文据实描述 |
| 其他情况（包括低于原文 66.4） | 定性图照常放 OOAL；图注给出复现数字。表 3 仍引用原文。不为追原文数字调参；如第 8 节有回复，再按作者给的设置重跑一次 |

复现数字低于原文较多时，读者会从图注看到差距，这是图注必须写复现数字的原因。

图注英文草稿（定稿时与全文一起润色）：

> OOAL predictions are from our re-implementation on UMD, since the UMD code is not publicly available; it obtains X / Y / Z mIoU (seen / unseen / hIoU), compared with the reported 74.6 / 59.7 / 66.4.

## 7. 统计 OOAL 差在哪里，再选图

**第 1 步：逐帧比较。** OOAL 评完之后即可运行；本方法默认读 `cm__reframe_bg`（指南中的主结果为 78.53 / 57.82）。用户提供的目标机目录如下，本机没有 `/home1` 挂载，尚未验证其中的文件；需先通过 `check_setup` 核对完整帧列表及结果版本。

```bash
R="${OURS_ROOT:-/home1/yhzheng/umd-one-shot/umd_eval_runs/reframe__tcrop__square_full/reframe_eval}"
python -m ooal_umd.check_setup --ours_root "$R"
python -m ooal_umd.compare_umd --split seen \
    --ooal $S/eval_020000/per_frame.npz --ours $R/seen/per_frame.npz --out figs/compare_seen
python -m ooal_umd.compare_umd --split unseen \
    --ooal $U/eval_020000/per_frame.npz --ours $R/unseen/per_frame.npz --out figs/compare_unseen
```

`report.md` 的内容：

| 节 | 内容 | 用来 |
|---|---|---|
| 1. 按类别 | 每类的 OOAL / 本方法 mIoU、差、本方法更好的帧占比、OOAL 三类错误的比例 | 定图里放哪几类：差大、且“本方法更好的帧”占比高的类别，说明差距是系统性的 |
| 2. 按可供性 | 两者的 IoU；OOAL 的 GT 像素被判对、判成背景、最常判成哪种可供性 | 看 OOAL 在哪种可供性上漏检多、和哪种可供性混淆 |
| 3. 候选帧：按类别 | 每类逐帧差最大的 3 帧，每个物体实例最多 1 帧 | 直接选图 |
| 4. 候选帧：按 OOAL 的主要错误 | 漏检、部件混淆、溢出三类各 10 帧 | 每种错误各选一例，图里各行展示不同的失败方式 |
| 5. 本方法更差的帧 | 差最小的 10 帧，附本方法的错误类型 | 局限一节的失败例 |

三类错误的定义（都除以该帧 GT 物体像素数，所以溢出可以超过 100%）：

- **漏检**：GT 部件像素被判成背景；
- **部件混淆**：GT 部件像素被判成另一种可供性（报告里给出最大的一对，如 scoop→support）；
- **溢出**：背景像素被判成某种可供性。

逐帧 mIoU 只对该帧 GT 或预测中出现的可供性取平均，预测了 GT 中没有的可供性记 0。UMD 是转台拍摄，相邻帧几乎一样，所以候选列表里每个物体实例最多 1 帧。全部帧的数字在 `frames.csv`，可以自己排序筛选。

**第 2 步：定帧。** 从 `report.md` 里挑，或直接用生成的列表（`ids_by_category.txt`、`ids_miss.txt`、`ids_confuse.txt`、`ids_spill.txt`、`ids_ours_worse.txt`），把选中的帧写进一个文件，如 `figs/umd_seen_ids.txt`。

**第 3 步：本方法为这些帧导出 `<id>.png`**（0–7，原图尺寸），放到一个目录里。OOAL 的 `pred/` 已经包含全部测试帧。

**第 4 步：画图。**

```bash
python -m ooal_umd.viz_umd --split seen --ids $(cat figs/umd_seen_ids.txt) \
    --method OOAL=$S/eval_020000/pred --method Ours=/path/to/ours/seen/pred \
    --out figs/umd_seen --save-panels
```

- 同一行的所有列用同一个裁剪框（GT 物体框外扩 25% 的正方形），只为显示；配色为色盲友好的 Okabe-Ito。
- `--save-panels` 另存每一格的 png，方便在 LaTeX 里自己排版。
- unseen 同理，换 `--split unseen`。
- 这些帧是按差距挑的，图注不写“随机选取”。正文可以引用 `report.md` 第 1、2 节的类别或可供性差距，说明图中的失败在整个测试集上普遍存在，而不只是这几帧。
- `viz_umd.py` 仍保留 `--per-category K --seed S` 的随机抽帧，附录如需不挑选的例子可以用它。

## 8. 并行可做的两件事（不阻塞上面）

- **给作者写信。** 你本来就打算问背景口径，可以一起问：UMD 的 checkpoint 或预测图是否还在，以及输入分辨率与背景的处理。代码丢了不代表权重和预测也丢了；有预测图就直接画，不用复现。
- **问 issue #12 的 cjiang2。** 他说已经复现了 UMD 接入，可以在那个 issue 下问他的设置与结果。

有回复就用来核对第 3 节的决定，不用等。

## 9. 文件

| 文件 | 作用 |
|---|---|
| `umd_protocol.py` | 导入 `umd_eval.py`，提供帧列表、标签、one-hot、预测到标签、hIoU（不依赖 torch） |
| `data_umd.py` | UMD 训练集（one-shot 示例）与测试集 |
| `model_umd.py` | `OOALUMD`（`Net` 加 `forward_all`）与自检 |
| `train_umd.py` | 训练，仓库默认超参，不做测试集选模型 |
| `test_umd.py` | 评测，写 `summary.json`、`per_frame.npz`、`pred/` |
| `report_umd.py` | seen / unseen / hIoU 与原文对照 |
| `compare_umd.py` | 逐帧比较 OOAL 与本方法：按类别、可供性、错误类型统计，输出候选帧（不依赖 torch） |
| `viz_umd.py` | 定性图（不依赖 torch） |
| `test_protocol_numpy.py` | 不依赖 torch 的测试（我这里已通过 14 项） |
| `test_torch_cpu.py` | 需要 torch 的 CPU 测试；2026-10-11 与 NumPy 测试合计 21 项在服务器通过，两种 DrawAfford 布局均通过 |
| `local_backbones.py` | 显式加载本地原生 DINOv2 和官方 CLIP 权重，避免构造时联网 |
| `check_setup.py` | 新服务器数据清单和本方法逐帧缓存核对 |
| `SERVER_HANDOFF.md` | 目标服务器安装、路径、冒烟、正式实验及可视化操作手册 |
