from collections import Counter
from copy import deepcopy
from functools import reduce
from typing import Dict

import torch
import torchvision.ops as torchops
from fastNLP import Vocabulary
from torch import Tensor
from torchmetrics import Metric

from src.utility.logger import get_logger_func

_warn, _info, _debug = get_logger_func("metric")
EPS = 1e-12


class DependencyParsingMetric(Metric):
    def __init__(self, extra_vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("correct_arcs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_rels", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.has_label = False

    def update(self, predict, gold, mask):
        arc_preds, arc_golds = predict["arc"], gold["arc"]
        arc_mask = arc_preds.eq(arc_golds) & mask
        arc_mask_seq = arc_mask[mask]

        self.n += len(mask)
        self.total += len(arc_mask_seq)

        lens = mask.sum(1)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.correct_arcs += arc_mask_seq.sum().item()

        if "rel" in predict:
            self.has_label = True
            rel_preds, rel_golds = predict["rel"], gold["rel"]
            rel_mask = rel_preds.eq(rel_golds) & arc_mask
            rel_mask_seq = rel_mask[mask]

            self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()
            self.correct_rels += rel_mask_seq.sum().item()

    def compute(self):
        _debug(
            f"sent: {self.n}, token: {self.total}, c_arc: {self.correct_arcs}, c_rel: {self.correct_rels}"
        )
        out = {
            "ucm": 100 * self.n_ucm / (self.n + EPS),
            "uas": 100 * self.correct_arcs / (self.total + EPS),
        }
        if self.has_label:
            out["lcm"] = 100 * self.n_lcm / (self.n + EPS)
            out["las"] = 100 * self.correct_rels / (self.total + EPS)
        return out


class FactorImageMatchingMetric(Metric):
    def __init__(self, extra_vocab):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predict, gold, mask):

        if "txt_to_img" not in predict:
            return

        txt_to_img = predict["txt_to_img"]

        self.total += sum(len(x) for x in txt_to_img)
        self.correct += sum(
            sum(item == i for item in x) for i, x in enumerate(txt_to_img)
        )

    def compute(self):
        return {"acc": 100 * self.correct / (self.total + 1e-6)}


class CaptionImageMatchingMetric(Metric):
    def __init__(self, extra_vocab):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predict, gold, mask):

        if "txt_to_img" not in predict:
            return

        txt_to_img = predict["txt_to_img"]

        self.total += len(txt_to_img)
        self.correct += sum(
            txt_to_img == torch.arange(len(txt_to_img), device=txt_to_img.device)
        )

    def compute(self):
        return {"acc": 100 * self.correct / (self.total + 1e-6)}


class BoxRelMatchingMetric(Metric):
    def __init__(self, extra_vocab):
        super().__init__()
        self.add_state("correct_obj", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_attr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_rel", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct_r_rel", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_obj", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_attr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_rel", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "processed_token", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, predict, gold, mask):
        if "sg_box" not in gold or gold["sg_box"].shape[2] == 0:
            # skip if no gold or no candidate for current batch
            return
        B, L, N, *_ = gold["sg_box"].shape

        # Batch[Sequence[data]], if obj/attr data is int, else (int, int)
        match = predict["txt_to_factor"]
        # [B, N, 4]
        proposal = gold["vis_box"]

        # [batch, token, n_candidate] # 1=obj, 2=attr, 3=rel
        gold_type = gold["sg_type"].unsqueeze(-1)
        # [batch, token, n_candidate] # 1=is an alignment
        gold_mask = gold["sg_mask"].unsqueeze(-1)
        # [batch, token, n_candidate, 2, 4] 4=(x,y,x,y) 2=pair,for obj/attr,[1] is useless
        gold_box: Tensor = gold["sg_box"].view(*gold["sg_box"].shape[:2], 1, 2, 4)

        max_num_predicted = max(
            len(token) for inst in predict["txt_to_factor"] for token in inst
        )
        pred_box = torch.zeros(B, L, max_num_predicted, 2, 4, device=proposal.device)
        pred_mask = torch.zeros(
            B, L, max_num_predicted, device=proposal.device, dtype=torch.bool
        )
        pred_type = torch.zeros(B, L, max_num_predicted, dtype=torch.int)
        seq_len = mask.sum(1).tolist()
        self.processed_token += mask.sum()
        for bid, inst_match in enumerate(match):
            pred_box_inst = []
            for tid, raw_token_match in enumerate(inst_match[: seq_len[bid]]):
                types = []
                token_match = []
                for type_, t in raw_token_match:
                    type_ = (
                        3
                        if type_ == "rel"
                        else 2
                        if type_ == "attr"
                        else 1
                        if type_ == "obj"
                        else 0
                    )
                    types.append(type_)
                    t = (t, t) if isinstance(t, int) else t
                    token_match.append(t)
                pred_type[bid, tid, : len(token_match)] = torch.tensor(types)
                pred_box_inst.append(proposal[bid, token_match])
            pred_box_inst = torch.stack(pred_box_inst, 0)
            pred_box[bid, : seq_len[bid], : len(pred_box_inst)] = pred_box_inst
            pred_mask[bid, : seq_len[bid], : len(pred_box_inst)] = 1
        mask = (pred_mask.unsqueeze(3) & gold_mask.unsqueeze(2)).unsqueeze(-1)
        pred_type = pred_type.to(pred_box.device)
        _raw = (_one_by_one_iou(pred_box, gold_box, dim=2) > 0.5) & mask
        obj_attr_iou = (_raw[..., 0] & (pred_type < 3).unsqueeze(-1)).view(
            B, L, -1
        ).any(-1) & ((gold_type[..., 0] > 0) & (pred_type[..., 0] > 0))
        rel_iou = (_raw.all(-1) & (pred_type == 3).unsqueeze(-1)).view(B, L, -1).any(-1)
        _raw2 = (
            _one_by_one_iou(pred_box, gold_box[:, :, :, [1, 0]], dim=2) > 0.5
        ) & mask
        rel_r_iou = (
            (_raw2.all(-1) & (pred_type == 3).unsqueeze(-1)).view(B, L, -1).any(-1)
        )

        self.correct_obj += ((gold_type[..., 0] == 1) & obj_attr_iou).sum()
        self.correct_attr += ((gold_type[..., 0] == 2) & obj_attr_iou).sum()
        self.correct_rel += ((gold_type[..., 0] == 3) & rel_iou).sum()
        self.correct_r_rel += ((gold_type[..., 0] == 3) & rel_r_iou).sum()
        self.total_obj += (gold_type[..., 0] == 1).sum()
        self.total_attr += (gold_type[..., 0] == 2).sum()
        self.total_rel += (gold_type[..., 0] == 3).sum()

    def compute(self):
        _debug(
            f"token: {self.processed_token} total_obj: {self.total_obj} total_attr: {self.total_attr}"
        )
        rel = max(self.correct_rel, self.correct_r_rel)
        out = {
            "acc": 100
            * (self.correct_obj + self.correct_attr + rel)
            / (self.total_obj + self.total_attr + self.total_rel + EPS),
            "obj": 100 * self.correct_obj / (self.total_obj + EPS),
            "attr": 100 * self.correct_attr / (self.total_attr + EPS),
            "rel": 100 * self.correct_rel / (self.total_rel + EPS),
        }
        return out


def _box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = torchops.boxes._upcast(boxes)
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def _one_by_one_iou(boxes1: Tensor, boxes2: Tensor, dim):
    """
    M, N is at dim=dim.
    :param boxes1: [..., N, ...X, 4]
    :param boxes2: [..., M, ...X, 4]
    :return: [..., N, M, ...X]
    """
    assert dim != -1, "The last dim must be the box."
    dim = boxes1.ndim - dim if dim < 0 else dim

    boardcast_shape = list(boxes1.shape)
    boardcast_shape.insert(dim + 1, boxes2.shape[dim])
    boxes1 = boxes1.unsqueeze(dim + 1).expand(boardcast_shape)
    boxes2 = boxes2.unsqueeze(dim).expand(boardcast_shape)
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)
    lt = torch.max(boxes1[..., :2], boxes2[..., :2])
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = torchops.boxes._upcast(rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter

    return (inter / union).view(boardcast_shape[:-1])


class MultiMetric(Metric):
    """combine different metrics"""

    def __init__(self, extra_vocab, **metric):
        super().__init__()
        self.metric = metric
        for name, metric in self.metric.items():
            self.add_module(name, metric)

    def update(self, predict, gold, mask):
        for m in self.metric.values():
            m.update(predict, gold, mask)

    def compute(self):
        out = {}
        for name, metric in self.metric.items():
            for key, value in metric.compute().items():
                if name == "main":
                    out[key] = value
                else:
                    out[f"{name}/{key}"] = value
        return out

    def reset(self):
        for m in self.metric.values():
            m.reset()

    def __hash__(self) -> int:
        return hash(tuple(self.children()))
