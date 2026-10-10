# OOAL UMD 复现实验：另一台服务器交接手册

更新时间：2026-10-11。目标仓库：<https://github.com/yhzhengoo/OOAL_umd>。

## 1. 当前状态与必须保留的协议

用户已要求停止在 `NF5280M6` 上运行，转到拥有 `/home1/yhzheng/umd-one-shot` 的服务器。当前机器没有 `/home1` 挂载，不能确认该目录中的结果文件。

本机已完成：

- 14 项 NumPy 测试和 7 项 torch CPU 测试，共 21 项全部通过、无 skipped；两种 DrawAfford 仓库布局均通过。
- 官方本地 DINOv2 / CLIP 构造成功，200 步冒烟 checkpoint 严格加载时全部参数匹配。在 CPU 的 28×42 小输入上，7 通道 `forward_all` 与原 `Net.forward` 的最大差为 0.0；CUDA 未初始化。这不替代目标机的完整尺寸 GPU 冒烟。
- 数据清单：Seen 54 张参考 / 14,020 张测试，Unseen 76 张参考 / 7,969 张测试；两组参考均覆盖 1–7 标签。
- 使用原版 OOAL 及官方 DINOv2、CLIP 权重完成 Seen 200 步 GPU 冒烟训练。
- `NF5280M6` 的所有 OOAL 训练/评测进程已停止。50 帧冒烟评测未完成，正式 20,000 步训练尚未启动，没有最终复现指标。

本机冒烟 checkpoint 保留在：
`/home/yhzheng/OOAL_umd/runs/smoke/seen/20261010_235557/iter_000200.pth`。
它不上传 GitHub、不作为正式实验的训练起点；目标服务器重新做冒烟，再从零开始两组正式训练。

固定协议来自 [OOAL_UMD复现实验.md](OOAL_UMD复现实验.md)：

- Seen、Unseen 各训练一次，分别使用对应的 54 / 76 张 one-shot 参考。
- SGD lr 0.01、momentum 0.9、Nesterov、weight decay 5e-4、cosine、batch 1、seed 321、20,000 iter。
- 训练 resize H×W = 476×644、random crop = 420×560、水平翻转；测试整帧 resize = 476×644，再将分数双线性恢复原图。
- 7 通道 one-hot 标签，背景全部为零，沿用原版 BCE。Ladle 的 contain→scoop 对参考和测试 GT 同样应用。
- 最大前景 sigmoid 分数严格 `> 0.5` 才输出 argmax+1，否则背景；不做通道 min-max，不使用查询类别门控。
- 汇总全部帧的 8×8 混淆矩阵后计算 7 前景类 mIoU，背景不参加宏平均。
- 每 5,000 步保存 checkpoint，只评最后的 `iter_020000.pth`；不按测试集结果选模型或调阈值。
- **无论复现数字高低，都画 OOAL；按逐帧统计选图，并标明选择方式和复现数字。**

## 2. 本次代码调整

`models/ooal.py` 没有修改，模型结构、训练分支、损失、参数名保持原版。

| 文件 | 调整 |
|---|---|
| `umd_protocol.py` | 自动支持两种 DrawAfford 目录布局；通过环境变量覆盖数据路径；验证 7 类/0.5 协议；记录实际评测源码路径与 SHA-256 |
| `local_backbones.py`、`model_umd.py` | 构造时可显式加载原生 DINOv2 仓库和官方权重、OpenAI CLIP 权重；构造结束即恢复加载器，forward 不变 |
| `train_umd.py`、`test_umd.py` | 新增本地权重参数，输出中记录实际数据路径、评测源码校验值、torch 版本 |
| `check_setup.py` | 不依赖 torch，核对训练/测试清单及本方法逐帧结果的 ID、对象名、键、矩阵形状和像素数 |
| `requirements-server.txt` | 隔离环境依赖；固定 setuptools 69.5.1，避免 CLIP 的 pkg_resources 导入失败 |
| `.gitignore` | 排除 `.runtime/` 中的权重、虚拟环境、运行结果和临时缓存 |

原版 `torch.hub.load` 即使已有缓存，也可能先联网查询默认分支。本地加载选项用 `source='local', pretrained=False` 构造 DINO，再严格加载官方 state_dict，避免联网查询及旧 `hubconf.py` 中硬编码权重路径。不要将 Hugging Face 的 `pytorch_model.bin` 直接作为官方 DINO checkpoint。

## 3. 获取代码与环境

在目标服务器克隆 fork。若环境代理不可用，可仅对 clone 命令取消代理；不用修改全局代理。

```bash
git clone https://github.com/yhzhengoo/OOAL_umd
cd OOAL_umd
```

已有 torch / torchvision 配套环境时，在该环境内创建隔离 venv：

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r ooal_umd/requirements-server.txt
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())'
```

本机验证组合：Python 3.10、torch 2.6.0+cu124、torchvision 0.21.0+cu124、NumPy 1.26.4、setuptools 69.5.1。如果另建环境，可安装这一匹配组合：

```bash
conda create -n ooal-umd python=3.10 pip -y
conda activate ooal-umd
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r ooal_umd/requirements-server.txt
```

该 CUDA wheel 要求目标机驱动支持 CUDA 12.4。使用其他 torch 版本时，torchvision 也必须匹配。以下命令统一使用已激活环境中的 `python`。

## 4. 配置目标服务器路径

`DRAWAFFORD_ROOT` 指向**代码仓库根目录**，不是 `reframe_eval` 结果目录。支持：

```text
<DRAWAFFORD_ROOT>/affordance_research/umd_eval/umd_eval.py  # ConceptAttention 布局
<DRAWAFFORD_ROOT>/umd_eval/umd_eval.py                      # 独立 drawafford_aed 布局
```

必须保留同仓库的 `afbank/one_shot`、`dino_transfer` 等依赖；不能只复制一个 `umd_eval.py`。

按目标服务器实际位置替换第一个变量；后三项对应另一版 DrawAfford 的 `/home1` 默认配置，运行前检查确实存在。

```bash
export DRAWAFFORD_ROOT=/实际位置/drawafford_aed
export UMD_DATASET_ROOT=/home1/yhzheng/part-affordance-dataset
export UMD_BASE_REF_ROOT=/home1/yhzheng/umd-one-shot/train_base_one_pick
export UMD_NOVEL_REF_ROOT=/home1/yhzheng/umd-one-shot/train_novel_one_pick
export OURS_ROOT=/home1/yhzheng/umd-one-shot/umd_eval_runs/reframe__tcrop__square_full/reframe_eval
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

每个新终端都需重新设置这些变量。路径覆盖只作用于本进程中的 evaluator，不改写 DrawAfford 原代码或 UMD 数据。

## 5. 本地官方权重

可以复用目标机已有的**官方**文件；也可以准备在 fork 的 `.runtime/`，该目录不会提交。

```bash
mkdir -p .runtime/checkpoints
git clone https://github.com/facebookresearch/dinov2.git .runtime/dinov2
git -C .runtime/dinov2 checkout c3c2683a13cde94d4d99f523cf4170384b00c34c
curl -fL --retry 2 -o .runtime/checkpoints/dinov2_vitb14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
curl -fL --retry 2 -o .runtime/checkpoints/ViT-B-16.pt \
  https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
sha256sum .runtime/checkpoints/*.p*
```

本机官方文件的 SHA-256：

```text
dinov2_vitb14_pretrain.pth  0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
ViT-B-16.pt                5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
```

```bash
export OOAL_DINO_REPO="$PWD/.runtime/dinov2"
export OOAL_DINO_WEIGHTS="$PWD/.runtime/checkpoints/dinov2_vitb14_pretrain.pth"
export OOAL_CLIP_WEIGHTS="$PWD/.runtime/checkpoints/ViT-B-16.pt"
```

三个环境变量也可改用 CLI `--dino_repo`、`--dino_weights`、`--clip_weights`。DINO 的两个变量需成对设置；没有这些变量时仍走原版联网加载。CLIP 使用原仓库的 tokenizer 与 `build_model`，不是 HF 文本适配器。

## 6. 测试与目录确认

```bash
python -m unittest ooal_umd.test_protocol_numpy ooal_umd.test_torch_cpu -v
python -m ooal_umd.check_setup --ours_root "$OURS_ROOT"
```

预期测试 21 项全部通过，无 skipped。`check_setup` 核对：

- Seen 54 参考、14,020 测试；Unseen 76 参考、7,969 测试。
- `$OURS_ROOT/{seen,unseen}/per_frame.npz` 含 `ids`、`objects`、`cm__reframe_bg`。
- IDs 与全量 UMD split 一一对应，对象名一致，每帧矩阵为 8×8，累积像素数为 480×640。
- 输出本方法的逐类 IoU 与 mIoU；指南中的主结果约为 Seen 78.53%、Unseen 57.82%。如不一致，先查结果版本和矩阵键，不能直接当成这组主结果。

缺本方法结果时先运行 `python -m ooal_umd.check_setup`，OOAL 的训练与评测可以继续，但逐帧比较需要上述两个文件。

## 7. 冒烟：200 步 + 50 帧

```bash
python -u -m ooal_umd.train_umd --split seen --gpu 0 \
  --iters 200 --save_step 200 --show_step 20 --save_root runs/smoke
SMOKE_CKPT="$(find runs/smoke/seen -name iter_000200.pth | sort | tail -n 1)"
python -u -m ooal_umd.test_umd --split seen --gpu 0 --limit 50 \
  --model_file "$SMOKE_CKPT" --out_dir runs/smoke/eval
python -c 'import json; from pathlib import Path; p=Path("runs/smoke/eval"); s=json.loads((p/"summary.json").read_text()); assert s["frames"]==50 and s["self_check_max_diff"]<1e-5; assert len(list((p/"pred").glob("*.png")))==50; print("smoke passed")'
```

要求真实模型第一帧 `forward_all` 与原 `Net.forward` 逐通道一致（最大差 < 1e-5），且有 50 张原图尺寸的标签 PNG。200 步的性能只检查流程，不用于与原文比较。

## 8. 正式训练与全量评测

单 GPU 串行运行，两组均从零开始。多 GPU 时可将 Unseen 的 `--gpu` 改为实际存在的另一张卡。

```bash
python -u -m ooal_umd.train_umd --split seen --gpu 0
python -u -m ooal_umd.train_umd --split unseen --gpu 0
```

每次新训练创建 `runs/ooal_umd/<split>/<时间>/`，在 `config.json` 和 `run.log` 留下记录。本脚本没有 resume 参数；中断后直接重跑会新建实验，不会自动续训。

根据日志将 `S`、`U` 设置为**本次正式训练**目录：

```bash
export S=runs/ooal_umd/seen/实际时间
export U=runs/ooal_umd/unseen/实际时间
python -u -m ooal_umd.test_umd --split seen --gpu 0 \
  --model_file "$S/iter_020000.pth" --out_dir "$S/eval_020000"
python -u -m ooal_umd.test_umd --split unseen --gpu 0 \
  --model_file "$U/iter_020000.pth" --out_dir "$U/eval_020000"
python -m ooal_umd.report_umd --seen "$S/eval_020000" --unseen "$U/eval_020000" \
  --out ooal_umd_report.md
```

正式测试不传 `--limit`、`--no_save_pred`。完成后检查 Seen 14,020 / Unseen 7,969 帧及同数量 PNG；输出均包含 `summary.json`、`per_frame.npz`、`pred/`。每张 PNG 为原图 480×640、单通道标签 0–7。

## 9. 统计、选帧、可视化（数字高低均执行）

```bash
python -m ooal_umd.compare_umd --split seen \
  --ooal "$S/eval_020000/per_frame.npz" --ours "$OURS_ROOT/seen/per_frame.npz" \
  --out figs/compare_seen
python -m ooal_umd.compare_umd --split unseen \
  --ooal "$U/eval_020000/per_frame.npz" --ours "$OURS_ROOT/unseen/per_frame.npz" \
  --out figs/compare_unseen
```

默认本方法键为 `cm__reframe_bg`。先阅读两个 `report.md` 的类别和可供性统计，再用 `ids_by_category.txt`、`ids_miss.txt`、`ids_confuse.txt`、`ids_spill.txt` 选图；保留 `ids_ours_worse.txt` 中的失败例供局限部分使用。每类错误选择有统计支持的实例，避免多选同一转台对象的邻帧。

OOAL 的预测已齐全，可以先画 OOAL。下面将类别候选列表的前 6 张画成 Image / GT / OOAL，同时保存独立面板：

```bash
if test -s figs/compare_seen/ids_by_category.txt; then
  mapfile -t IDS < <(head -n 6 figs/compare_seen/ids_by_category.txt)
  python -m ooal_umd.viz_umd --split seen --ids "${IDS[@]}" \
    --method "OOAL=$S/eval_020000/pred" --out figs/umd_seen_ooal --save-panels
fi
```

若本方法没有更好帧，不能因而跳过 OOAL 可视化：根据 `frames.csv` 明确选择帧（包括本方法更差的帧），再传入 `--ids`。不要将空候选列表传给 `--ids`，因为空列表会回退到随机抽帧。

对这些确定的 ID，从本方法原实验缓存/导出工具生成 0–7、480×640 的标签 PNG。`per_frame.npz` 只有混淆矩阵，**不能反推出空间预测图**。本 fork 不包含 DrawAfford 缓存重建器，需使用那台服务器的实际导出代码；不可从 GT 或混淆矩阵伪造预测。

准备好本方法标签图目录后：

```bash
export OURS_SEEN_PRED=/实际位置/ours/seen/pred
python -m ooal_umd.viz_umd --split seen --ids "${IDS[@]}" \
  --method "OOAL=$S/eval_020000/pred" --method "Ours=$OURS_SEEN_PRED" \
  --out figs/umd_seen --save-panels
```

Unseen 同理，使用 `figs/compare_unseen`、`$U/eval_020000/pred` 和 Unseen 的本方法标签图目录。所有方法使用同一 GT 裁剪框、同一配色；输出 PNG、PDF、选中 IDs 与独立面板。

图注报告实际 Seen / Unseen / hIoU，并说明按性能差和错误类型选帧，不称为随机例子。是否向表 3 添加复现行，按原指南第 6 节处理；无需为了追原文数字再次调参。完成后将环境、最终目录、逐类 IoU、选图理由记录进本地《实验细节_本地保留》。

## 10. 验收与故障定位

- `pkg_resources` 缺失或 `packaging` 导入失败：确认运行的是隔离环境，且 setuptools 为 69.5.1；不要只在其他环境安装。
- 网络代理 `127.0.0.1:17890` 失效：对需要下载的命令使用 `env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy ...`；训练/测试显式使用本地权重。
- `DRAWAFFORD_ROOT` 找不到：核对上面两种仓库布局，结果目录不等于代码目录。
- 清单数或逐帧 ID 不符：核对三个 UMD 路径及本方法结果版本，不通过自检时停止正式比较。
- DINO `strict=True` 加载失败：检查是不是官方 ViT-B/14 权重，以及本地代码是否混入其他模型改动。
- 单 GPU 不要照搬原指南的 `--gpu 1`；用 `nvidia-smi` 确认实际卡号。
- 当前机器的冒烟 checkpoint 和虚拟环境均未上传 GitHub。交接的是代码与协议，正式结果需要在目标服务器重新生成。
