from itertools import chain
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from fastNLP.core import DataSet
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from src.datamodule.task.dep import DepDataModule
from src.utility.logger import get_logger_func

InputDict = Dict[str, Tensor]
TensorDict = Dict[str, Tensor]
AnyDict = Dict[str, Any]
GenDict = (dict, DictConfig)
GenList = (list, ListConfig)

_warn, _info, _debug = get_logger_func("datamodule")


def get_box(obj):
    return [obj["x"], obj["y"], obj["x"] + obj["width"], obj["y"] + obj["height"]]


class _COCODetFeatLazyLoader:
    def __init__(self, root, sg_data, sample, gold):
        self.root = root
        self.sg_data = sg_data
        self.sample = sample
        self.gold = gold

    def __call__(self, batch: List[Tuple[int, Any]]):
        box_feats, boxes, masks, rel_masks = [], [], [], []
        max_len = 0
        for _, inst in batch:
            if (self.root / f"{inst['img_id']}.npy").exists():
                feat = np.load(str(self.root / f"{inst['img_id']}.npy"))
                if self.sample > 0 and self.sample < len(feat):
                    sample_id = np.random.choice(
                        np.arange(len(feat)), self.sample, False
                    )
                    feat = feat[sample_id]
                else:
                    feat = feat[:35]
                    sample_id = np.arange(len(feat))
                box_feat, box = feat[:, :-4], feat[:, -4:]
                box_feat = torch.tensor(box_feat, dtype=torch.float)
                box = torch.tensor(box)

                box_feats.append(box_feat)
                boxes.append(box)

                if self.gold:
                    inst_mask, inst_rel_mask = self.build_gold_mask(inst, sample_id)
                    masks.append(inst_mask)
                    rel_masks.append(inst_rel_mask)
                else:
                    masks.append(torch.ones(len(box_feat), dtype=torch.bool))
                    rel_masks.append(None)
                max_len = max(len(box_feat), max_len)
            else:
                assert False

        box_feats_output = torch.zeros(len(box_feats), max_len, 2048)
        boxes_output = torch.zeros(len(boxes), max_len, 4)
        masks_output = torch.zeros(len(masks), max_len, dtype=torch.bool)
        rel_masks_output = (
            None
            if len(rel_masks) == 0
            else torch.zeros(len(rel_masks), max_len, max_len, dtype=torch.bool)
        )
        for i, (bf, b, m, rm) in enumerate(zip(box_feats, boxes, masks, rel_masks)):
            if bf is not None:
                box_feats_output[i, : len(bf)] = bf
                boxes_output[i, : len(b)] = b
                masks_output[i, : len(m)] = m
                if rm is not None:
                    rel_masks_output[i, : rm.shape[0], : rm.shape[1]] = rm

        return (
            {
                "vis_box_feat": box_feats_output,
                "vis_box_mask": masks_output,
                "vis_rel_mask": rel_masks_output,
                "vis_available": masks_output[:, 0],
            },
            {"vis_box": boxes_output},
        )

    def build_gold_mask(self, inst, sample_id):
        sg_inst = self.sg_data[inst["img_id"]]
        if len(sg_inst["obj"]) == 0:
            return torch.zeros(0, dtype=torch.bool), torch.zeros(0, 0, dtype=torch.bool)
        mask = torch.ones(min(len(sample_id), len(sg_inst["obj"])), dtype=torch.bool)
        rel_mask = torch.zeros(
            len(sg_inst["obj"]), len(sg_inst["obj"]), dtype=torch.bool
        )
        for item in sg_inst["rel"]:
            rel_mask[item["subj"], item["obj"]] = 1
        sample_id = torch.from_numpy(sample_id)
        rel_mask = rel_mask.gather(
            1, sample_id.unsqueeze(0).expand(rel_mask.shape[1], -1)
        ).gather(0, sample_id.unsqueeze(-1).expand(-1, len(sample_id)))
        return mask, rel_mask


class VLParseDataModule(DepDataModule):
    TARGETS = ("arc", "sg_type", "sg_box", "sg_mask")
    # train: text(.conll), proposed box(det_feats/.npy), img(.npy)
    # dev: text(.conll), proposed box(det_feats/.npy), img(.npy), scene graph(../.json)
    # test: text(.conll), proposed box(det_feats/.npy), scene graph(../.json)

    def __init__(self, use_img, use_gold_scene_graph, sg_path, **kwargs):

        self.use_img = use_img  # use native image feature
        if self.use_img:
            self.INPUTS = self.INPUTS + ("vis_img",)
        self.use_gold_scene_graph = use_gold_scene_graph  # return gold box and rels

        with open(sg_path) as f:  # load scene graph
            sg_data = json.load(f)
            self.sg_data = {inst["coco_id"]: inst for inst in sg_data}

        if use_gold_scene_graph:
            with open(os.path.split(sg_path)[0] + "/vlparse_train_sg_raw.json") as f:
                sg_data = json.load(f)
                self.sg_data |= {inst["coco_id"]: inst for inst in sg_data}

        super().__init__(**kwargs)

    def _load(self, path, name) -> DataSet:
        # text: xxx.conll, a conllu format file
        # img: xxx.npy, each item is prefeteched feat. [n_img x hidden_size]
        # det_feats/<id>.npy, box feat for each img  shape: 100 x (1024+4)
        # id_list/xxx.txt, each line is a img_id and sent_id pair. assume sent with same img_id are put together.
        ds: DataSet = super()._load(path + ".conll", name)

        # load ids
        folder, filename = os.path.split(path)
        with open(Path(folder) / "id_list" / (filename + ".txt")) as f:
            img_id = [int(line.strip()) for line in f]
            if len(img_id) != len(ds):
                img_id = [id_ for id_ in img_id for _ in range(5)]
        ds.add_field("img_id", img_id)
        ds.add_field("img_sent_id", [i % 5 for i, _ in enumerate(img_id)])

        # native image feature
        with self.tolerant_exception(["test"], name):
            if self.use_img:
                img_feat = np.load(path + ".npy").repeat(5, 0)
                ds.add_field("vis_img", img_feat, is_input=True)

        # prepare target, (and input if gold_sg) from sg data
        ds.apply_more(self.process_sg)

        ds.add_collate_fn(
            _COCODetFeatLazyLoader(
                Path(folder)
                / ("gold_feats" if self.use_gold_scene_graph else "det_feats"),
                self.sg_data,
                35 if name in ("train", "train_init") else 0,
                self.use_gold_scene_graph,
            ),
            "det_feat_loader",
        )
        if name in ("dev", "test") or self.use_gold_scene_graph:
            ds.drop(lambda x: not x["has_sg"])
        return ds

    def process_sg(self, inst):
        if inst["img_id"] not in self.sg_data:
            txt2sg = {}
            rels = []
        else:
            sg = self.sg_data[inst["img_id"]]
            rels = sg["rel"]
            txt2sg = sg["txt2sg"][inst["img_sent_id"]]
            id2node = {node["id"]: node for node in chain(sg["obj"], sg["rel"])}
        typestr2id = {"OBJ": 1, "ATTR": 2, "REL": 3}
        gold_box, tok_type = [], []

        # here only collect grounded box per words
        for i in range(len(inst["raw_word"])):
            if (i := str(i)) in txt2sg:
                alignment = txt2sg[i]
                tok_type.append(typestr2id[alignment["type"]])
                if tok_type[-1] == 3:
                    node = id2node[alignment["preferred"]]
                    subj, obj = id2node[node["subj"]], id2node[node["obj"]]
                    gold_box.append(get_box(subj) + get_box(obj))
                else:
                    gold_box.append(
                        get_box(id2node[alignment["preferred"]]) + [0.0] * 4
                    )
            else:
                tok_type.append(0)
                gold_box.append([0.0] * 8)

        sg_rel = [[item["subj"], item["obj"]] for item in rels]
        return {
            "sg_type": tok_type,
            "sg_box": gold_box,
            "vis_rel": sg_rel,  # this is for inputs. When eval we just need sg_box.
            "sg_mask": [t != 0 for t in tok_type],
            "has_sg": inst["img_id"] in self.sg_data,
        }
