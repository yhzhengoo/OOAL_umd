"""Where OOAL does worse than ours on UMD: per-frame statistics for picking figure frames.

No torch.  Both inputs are per-frame confusion matrices on the same test frames:

    DRAWAFFORD_ROOT=/path/to/ConceptAttention \
    python -m ooal_umd.compare_umd --split seen \
        --ooal $S/eval_020000/per_frame.npz \
        --ours $RUNS/full_cached/reframe__tcrop__square_full/reframe_eval/seen/per_frame.npz \
        --out  figs/compare_seen

``--ours-key`` defaults to ``cm__reframe_bg`` (the main result, 78.53 / 57.82).

Outputs in ``--out``:
    report.md                 tables + candidate frames
    frames.csv                one row per frame (sort / filter it yourself)
    ids_by_category.txt       candidates, best category gap first
    ids_miss.txt / ids_confuse.txt / ids_spill.txt
                              candidates whose main OOAL error is that type
    ids_ours_worse.txt        frames where ours is clearly worse (limitations)

Definitions (rows of a confusion matrix = GT, columns = prediction, 0 = background):
    frame mIoU   mean IoU over the affordances in GT or prediction of that frame
    miss         GT affordance pixels predicted as background      / GT object pixels
    confuse      GT affordance pixels predicted as another one     / GT object pixels
    spill        background pixels predicted as an affordance      / GT object pixels
Category / affordance IoUs are computed from confusion matrices summed over
frames, like the split mIoU.  Candidate lists keep at most one frame per
object instance, because neighbouring turntable frames look almost the same.
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from collections import defaultdict

import numpy as np

from ooal_umd import umd_protocol as P

NAMES = ("grasp", "cut", "scoop", "contain", "pound", "support", "wrap-grasp")
ERRORS = ("miss", "confuse", "spill")
ERROR_ZH = {"miss": "漏检（部件判成背景）", "confuse": "部件混淆（判成别的可供性）",
            "spill": "溢出（背景判成部件）"}


# ---------------------------------------------------------------- loading
def load(path: str, key: str):
    d = np.load(path, allow_pickle=False)
    if key not in d.files:
        raise SystemExit(f"{path}: no '{key}', available {d.files}")
    cm = d[key].astype(np.int64)
    if cm.ndim != 3 or cm.shape[1:] != (8, 8):
        raise SystemExit(f"{path}:{key} has shape {cm.shape}, expected (N, 8, 8)")
    return np.asarray(d["ids"]).astype(str), np.asarray(d["objects"]).astype(str), cm


def align(ooal, ours):
    """Reorder ``ours`` to OOAL's frame order; both must cover the same frames."""
    ids_a, obj_a, cm_a = ooal
    ids_b, obj_b, cm_b = ours
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b):
        raise SystemExit("duplicate frame ids")
    if set(ids_a) != set(ids_b):
        only_a, only_b = sorted(set(ids_a) - set(ids_b)), sorted(set(ids_b) - set(ids_a))
        raise SystemExit(f"frame sets differ: {len(only_a)} only in OOAL {only_a[:3]}, "
                         f"{len(only_b)} only in ours {only_b[:3]}")
    pos = {i: k for k, i in enumerate(ids_b)}
    order = np.array([pos[i] for i in ids_a])
    if not np.array_equal(obj_b[order], obj_a):
        raise SystemExit("object names differ for the same frame ids")
    return ids_a, obj_a, cm_a, cm_b[order]


# ---------------------------------------------------------------- metrics
def class_iou(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cm (..., 8, 8) -> IoU (..., 7) of the affordances and their GT pixel counts."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)[..., 1:].astype(np.float64)
    gt = cm.sum(-1)[..., 1:].astype(np.float64)
    pred = cm.sum(-2)[..., 1:].astype(np.float64)
    union = gt + pred - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    return iou, gt


def frame_miou(cm: np.ndarray) -> np.ndarray:
    """Mean IoU over affordances in GT or prediction; a predicted affordance absent from GT counts as 0."""
    iou, _ = class_iou(cm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # frames with no affordance at all -> NaN
        return np.nanmean(iou, axis=-1)


def error_fractions(cm: np.ndarray) -> np.ndarray:
    """(N, 8, 8) -> (N, 3): miss, confuse, spill, each divided by GT object pixels."""
    fg = cm[:, 1:, :].sum((1, 2)).astype(np.float64)
    correct = np.trace(cm[:, 1:, 1:], axis1=1, axis2=2)
    miss = cm[:, 1:, 0].sum(1)
    confuse = cm[:, 1:, 1:].sum((1, 2)) - correct
    spill = cm[:, 0, 1:].sum(1)
    errs = np.stack([miss, confuse, spill], 1).astype(np.float64)
    return np.divide(errs, fg[:, None], out=np.zeros_like(errs), where=fg[:, None] > 0)


def top_wrong_pair(cm: np.ndarray) -> list[str]:
    """Per frame, the largest GT->pred confusion between two different affordances."""
    off = cm[:, 1:, 1:].copy()
    idx = np.arange(7)
    off[:, idx, idx] = 0
    flat = off.reshape(len(cm), -1)
    best = flat.argmax(1)
    out = []
    for k, b in enumerate(best):
        out.append("" if flat[k, b] == 0 else f"{NAMES[b // 7]}→{NAMES[b % 7]}")
    return out


def split_miou(cm_sum: np.ndarray) -> float:
    iou, _ = class_iou(cm_sum)
    return float(np.nan_to_num(iou).mean())    # same as umd_eval.scores: all 7 affordances


def present_miou(cm_sum: np.ndarray) -> float:
    """mIoU over the affordances that occur in the GT of these frames (for one category)."""
    iou, gt = class_iou(cm_sum)
    keep = gt > 0
    return float(np.nan_to_num(iou[keep]).mean()) if keep.any() else float("nan")


# ---------------------------------------------------------------- selection
def one_per_instance(order, objects, limit):
    seen, picked = set(), []
    for k in order:
        if objects[k] in seen:
            continue
        seen.add(objects[k])
        picked.append(k)
        if len(picked) == limit:
            break
    return picked


def pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f}"


def frame_rows(idx, ids, objects, m_ooal, m_ours, err, pair):
    rows = ["| 帧 | OOAL | 本方法 | 差 | OOAL 主要错误 | 漏检 / 混淆 / 溢出 | 最大混淆 |",
            "|---|---:|---:|---:|---|---|---|"]
    for k in idx:
        main = ERRORS[int(err[k].argmax())] if err[k].max() > 0 else "—"
        rows.append(f"| `{ids[k]}` | {pct(m_ooal[k])} | {pct(m_ours[k])} | "
                    f"{100 * (m_ours[k] - m_ooal[k]):+.1f} | {main} | "
                    f"{pct(err[k, 0])} / {pct(err[k, 1])} / {pct(err[k, 2])} | {pair[k]} |")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("seen", "unseen"), required=True)
    p.add_argument("--ooal", required=True, help="OOAL per_frame.npz from test_umd")
    p.add_argument("--ooal-key", default="cm")
    p.add_argument("--ours", required=True, help="our per_frame.npz")
    p.add_argument("--ours-key", default="cm__reframe_bg")
    p.add_argument("--per-category", type=int, default=3, help="candidates per category")
    p.add_argument("--per-error", type=int, default=10, help="candidates per OOAL error type")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ids, objects, cm_o, cm_u = align(load(args.ooal, args.ooal_key), load(args.ours, args.ours_key))
    cats = np.array([o.split("_", 1)[0] for o in objects])
    m_o, m_u = frame_miou(cm_o), frame_miou(cm_u)
    gap = m_u - m_o
    err_o, err_u = error_fractions(cm_o), error_fractions(cm_u)
    pair_o, pair_u = top_wrong_pair(cm_o), top_wrong_pair(cm_u)
    os.makedirs(args.out, exist_ok=True)

    # sanity: split mIoU from the summed matrices, as umd_eval.scores computes it
    ev = P.umd_eval()
    s_o = ev.scores(cm_o.sum(0))["mIoU_7fg"]
    s_u = ev.scores(cm_u.sum(0))["mIoU_7fg"]
    assert abs(s_o - split_miou(cm_o.sum(0))) < 1e-9 and abs(s_u - split_miou(cm_u.sum(0))) < 1e-9

    lines = [f"# OOAL 与本方法在 UMD {args.split} 上的逐帧比较", "",
             f"- 帧数 {len(ids)}，物体实例 {len(set(objects))}，类别 {len(set(cats))}。",
             f"- 整体 mIoU（7 种可供性，与 `umd_eval.scores` 一致）：OOAL {pct(s_o)}，本方法 {pct(s_u)}。",
             f"- 逐帧 mIoU 本方法更高的帧占 {pct(float((gap > 0).mean()))}%；"
             f"高 10 个点以上 {pct(float((gap > 0.1).mean()))}%，低 10 个点以上 {pct(float((gap < -0.1).mean()))}%。",
             f"- 本方法：`{args.ours}` 的 `{args.ours_key}`；OOAL：`{args.ooal}` 的 `{args.ooal_key}`。", ""]

    # --- per category
    cat_stats = []
    for c in sorted(set(cats)):
        sel = cats == c
        a, b = present_miou(cm_o[sel].sum(0)), present_miou(cm_u[sel].sum(0))
        e = error_fractions(cm_o[sel].sum(0, keepdims=True))[0]
        aff = [NAMES[i] for i in np.nonzero(cm_o[sel].sum(0)[1:, :].sum(1))[0]]
        cat_stats.append((c, int(sel.sum()), len(set(objects[sel])), a, b, b - a,
                          float((gap[sel] > 0).mean()), e, aff))
    cat_stats.sort(key=lambda r: -r[5])
    lines += ["## 1. 按类别（差 = 本方法 − OOAL，从大到小）", "",
              "类别 mIoU 只对该类 GT 中出现的可供性取平均。", "",
              "| 类别 | 帧 | 实例 | 可供性 | OOAL | 本方法 | 差 | 本方法更好的帧 | OOAL 漏检 / 混淆 / 溢出 |",
              "|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for c, n, ni, a, b, d, win, e, aff in cat_stats:
        lines.append(f"| {c} | {n} | {ni} | {', '.join(aff)} | {pct(a)} | {pct(b)} | {100 * d:+.1f} | "
                     f"{pct(win)}% | {pct(e[0])} / {pct(e[1])} / {pct(e[2])} |")

    # --- per affordance, and where OOAL's GT pixels go
    iou_o, gt_px = class_iou(cm_o.sum(0))
    iou_u, _ = class_iou(cm_u.sum(0))
    rows_o = cm_o.sum(0)[1:, :].astype(np.float64)
    lines += ["", "## 2. 按可供性", "",
              "“OOAL 的 GT 像素去向”：该可供性的 GT 像素中，被 OOAL 判对、判成背景、判成其他可供性的比例。", "",
              "| 可供性 | OOAL IoU | 本方法 IoU | 差 | OOAL 判对 | 判成背景 | 最常判成 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for i, n in enumerate(NAMES):
        if gt_px[i] == 0:
            continue
        row = rows_o[i] / rows_o[i].sum()
        other = row[1:].copy()
        other[i] = 0
        j = int(other.argmax())
        wrong = f"{NAMES[j]} {pct(other[j])}%" if other[j] > 0 else "—"
        lines.append(f"| {n} | {pct(iou_o[i])} | {pct(iou_u[i])} | {100 * (iou_u[i] - iou_o[i]):+.1f} | "
                     f"{pct(row[i + 1])}% | {pct(row[0])}% | {wrong} |")

    # --- candidates per category
    by_cat = []
    lines += ["", "## 3. 候选帧：按类别", "",
              f"类别按第 1 节的差排序；每类按逐帧差从大到小取 {args.per_category} 帧，每个物体实例最多 1 帧。", ""]
    for c, *_ in cat_stats:
        sel = np.nonzero(cats == c)[0]
        order = sel[np.argsort(-gap[sel], kind="stable")]
        picked = [k for k in one_per_instance(order, objects, args.per_category) if gap[k] > 0]
        if not picked:
            continue
        by_cat += picked
        lines += [f"**{c}**", ""] + frame_rows(picked, ids, objects, m_o, m_u, err_o, pair_o) + [""]

    # --- candidates per OOAL error type
    main_err = np.where(err_o.max(1) > 0, err_o.argmax(1), -1)
    by_err = {}
    lines += ["## 4. 候选帧：按 OOAL 的主要错误", "",
              f"OOAL 在该帧的主要错误为此类、且本方法更好；按差从大到小，每个实例最多 1 帧，取 {args.per_error} 帧。", ""]
    for t, name in enumerate(ERRORS):
        sel = np.nonzero((main_err == t) & (gap > 0))[0]
        order = sel[np.argsort(-gap[sel], kind="stable")]
        picked = one_per_instance(order, objects, args.per_error)
        by_err[name] = picked
        lines += [f"**{ERROR_ZH[name]}**（OOAL 以此为主要错误的帧：{int((main_err == t).sum())}）", ""]
        lines += frame_rows(picked, ids, objects, m_o, m_u, err_o, pair_o) if picked else ["无"]
        lines.append("")

    # --- reverse: where ours is worse (limitations)
    sel = np.nonzero(gap < 0)[0]
    worse = one_per_instance(sel[np.argsort(gap[sel], kind="stable")], objects, args.per_error)
    lines += ["## 5. 本方法更差的帧（可用于局限）", "",
              "按差从小到大；“主要错误”一列此处是本方法的错误。", ""]
    lines += frame_rows(worse, ids, objects, m_o, m_u, err_u, pair_u) if worse else ["无"]
    lines.append("")

    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    def write_ids(name, idx):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write("".join(f"{ids[k]}\n" for k in idx))

    write_ids("ids_by_category.txt", by_cat)
    for name, idx in by_err.items():
        write_ids(f"ids_{name}.txt", idx)
    write_ids("ids_ours_worse.txt", worse)

    with open(os.path.join(args.out, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "object", "category", "ooal_miou", "ours_miou", "gap",
                    "ooal_miss", "ooal_confuse", "ooal_spill", "ooal_main_error", "ooal_top_wrong_pair",
                    "ours_miss", "ours_confuse", "ours_spill", "ours_top_wrong_pair"])
        for k in range(len(ids)):
            w.writerow([ids[k], objects[k], cats[k], f"{m_o[k]:.4f}", f"{m_u[k]:.4f}", f"{gap[k]:.4f}",
                        *(f"{x:.4f}" for x in err_o[k]),
                        ERRORS[main_err[k]] if main_err[k] >= 0 else "", pair_o[k],
                        *(f"{x:.4f}" for x in err_u[k]), pair_u[k]])
    print(f"OOAL {pct(s_o)} vs ours {pct(s_u)} on {len(ids)} frames -> {args.out}/report.md")


if __name__ == "__main__":
    main()
