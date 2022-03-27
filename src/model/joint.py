from bisect import bisect_left
from dataclasses import dataclass
from io import IOBase
from itertools import accumulate
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from fastNLP import DataSet, Vocabulary
from hydra.conf import MISSING
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module

from src import INF
from src.datamodule import DataModule
from src.model.base import JointModelBase
from src.model.ldndmv import LDNDMVConfig, DiscriminativeNDMV
from src.model.torch_struct import DMV1o
from src.utility.config import Config
from src.utility.fn import filter_list
from src.utility.var_pool import VarPool

InputDict = Dict[str, Tensor]
TensorDict = Dict[str, Tensor]
AnyDict = Dict[str, Any]

OBJ_POS = ["NN", "NNS", "PRP", "NNP", "WDT", "WP", "NNPS"]
REL_POS = [
    "IN",
    "VBZ",
    "VBG",
    "VBN",
    "TO",
    "VB",
    "RB",
    "RP",
    "VBD",
    "CC",
    "VBP",
    "EX",
    "POS",
    "FW",
    "WRB",
    "MD",
    "RBR",
]
ATTR_POS = ["DT", "JJ", "CD", "PRP$", "JJR", "JJS", "PDT"]


@dataclass
class DependencyBoxRelConfig(Config):
    dep_model_cfg: LDNDMVConfig

    margin: float
    word_encoder: AnyDict

    # structure
    add_rel: bool
    add_attr: bool
    add_image: bool
    add_marginal: bool

    language_factor_mode: str  # word, word+dep
    visual_factor_mode: str  # box, box+rel
    visual_factor_cfg: AnyDict
    feat_fuse_mode: str  # none, attention
    feat_fuse_args: AnyDict
    gather_logit_mode: str  # simple, reduced
    gather_logit_args: AnyDict
    loss_grounding_mode: str  # factor|ce, factor|hinge, cap_img|ll, factor|ce
    loss_grounding_args: AnyDict
    decode_grounding_mode: str  # on_img, on_box, on_box+rel
    decode_grounding_args: AnyDict
    grounding_interpolation: float  # balance grounding and dep

    # passthrough
    init_method: str
    init_epoch: int

    # ============================= AUTO FIELDS =============================
    n_word: int = MISSING
    n_tag: int = MISSING
    n_token: int = MISSING


class DependencyBoxRel(JointModelBase):
    def __init__(self, **cfg):
        super().__init__()

        # noinspection PyTypeChecker
        self.cfg: DependencyBoxRelConfig = cfg
        self.vis_factor_names = ["obj"]
        self.dependency: Optional[DiscriminativeNDMV] = None
        self.word_encoder: Optional[Module] = None
        self.criteria: Optional[Module] = None

    def setup(self, dm: DataModule):
        cfg = super()._setup(
            dm, DependencyBoxRelConfig, allow_missing={"n_word", "n_tag"}
        )

        n_x = self.encoder.get_dim("x")
        self.word_encoder = instantiate(cfg.word_encoder, n_in=n_x)
        self.criteria = nn.CrossEntropyLoss()
        self.set_impl_in_group("lang_feat", self.cfg.language_factor_mode)
        self.set_impl_in_group("vis_feat", self.cfg.visual_factor_mode)
        self.set_impl_in_group("feat_fuse", self.cfg.feat_fuse_mode)
        self.set_impl_in_group("gather_logit", self.cfg.gather_logit_mode)
        self.set_impl_in_group("loss_grounding", self.cfg.loss_grounding_mode)
        self.set_impl_in_group("decode_grounding", self.cfg.decode_grounding_mode)
        self.set_impl_in_group(
            "format_factor_prediction", self.cfg.decode_grounding_mode
        )

        # if cfg.use_pos_prior:
        v = dm.vocabs["tag"]
        self.register_buffer("pos_for_obj", torch.tensor([v[t] for t in OBJ_POS]))
        self.register_buffer("pos_for_attr", torch.tensor([v[t] for t in ATTR_POS]))
        self.register_buffer("pos_for_rel", torch.tensor([v[t] for t in REL_POS]))

    # vis_feat ===============================================================

    def vis_feat(self, inputs: InputDict, encoded: TensorDict, vp: VarPool):
        ...

    def vis_feat_init(self):
        if self.cfg.add_rel:
            self.vis_factor_names.append("rel")
        if self.cfg.add_attr:
            self.vis_factor_names.append("attr")
        if self.cfg.add_image:
            self.vis_factor_names.append("img")
        cfg: DictConfig = self.cfg.visual_factor_cfg
        self.vis_mlp_pre_matching = nn.Linear(
            self.vis_encoder.get_dim(None), cfg.n_hidden, bias=False
        )

    @JointModelBase.add_impl_to_group("vis_feat", "unprune", "vis_feat_init")
    def vis_feat_unprune(
        self, inputs: InputDict, encoded: TensorDict, vp: VarPool, return_mid=False
    ):
        _box_mask = inputs["vis_box_mask"]
        feat = [encoded["box"]]
        mask = [_box_mask]
        split = [_box_mask.shape[1]]
        if self.cfg.add_rel:
            feat.append(encoded["rel"])
            if inputs.get("vis_rel_mask") is not None:
                rel_mask = _box_mask.unsqueeze(1) * _box_mask.unsqueeze(2)
                # rel_mask.diagonal(dim1=1, dim2=2).fill_(False)
                rel_mask = rel_mask.triu(1)
                rel_mask = rel_mask.view(vp.batch_size, -1)
                mask.append(rel_mask)
            else:
                mask.append(inputs["vis_rel_mask"].view(vp.batch_size, -1))
            split.append(_box_mask.shape[1] * _box_mask.shape[1])
        if self.cfg.add_attr:
            feat.append(encoded["attr"])
            mask.append(_box_mask)
            split.append(_box_mask.shape[1])
        if self.cfg.add_image:
            feat.append(encoded["box"].mean(1, keepdim=True))
            mask.append(
                torch.ones(
                    len(encoded["box"]),
                    1,
                    dtype=torch.bool,
                    device=encoded["box"].device,
                )
            )
            split.append(1)
        vis = _mid = torch.cat(feat, dim=1)
        vis = self.vis_mlp_pre_matching(vis).refine_names("A", "V", "Y")
        vis_mask = torch.cat(mask, dim=1).refine_names("A", "V")
        if return_mid:
            return vis, vis_mask, split, _mid
        return vis, vis_mask, split

    # lang_feat ==============================================================

    # txt:          B x N+1 x H, no mask applied
    # txt_margin:   B x (N+1)+..., 0 for invalid, root has 1, root->tag has valid, tag->root has 0

    def lang_feat(
        self,
        inputs: InputDict,
        language_encoded: TensorDict,
        lang_score: TensorDict,
        vp: VarPool,
    ):
        ...

    @JointModelBase.add_impl_to_group("lang_feat", "word")
    def lang_feat_word_only(
        self,
        inputs: InputDict,
        language_encoded: TensorDict,
        lang_score: TensorDict,
        vp: VarPool,
    ):
        B, L, H = language_encoded["x"].shape
        mask = torch.cat([vp.mask.new_zeros(B, 1), vp.mask], dim=1).refine_names(
            "B", "Q"
        )
        x: Tensor = language_encoded["x"]
        root = (
            x.masked_fill(~vp.mask.unsqueeze(2), 0).sum(1) / vp.seq_len.unsqueeze(1)
        ).unsqueeze(1)
        x = torch.cat([root, x], dim=1)
        word_repr = self.word_encoder(x).refine_names("B", "Q", "X")
        return word_repr, mask, mask.to(torch.float)

    # noinspection PyAttributeOutsideInit
    def lang_feat_arc_mlp_init(self):
        self.child_encoder = instantiate(
            self.cfg.word_encoder, n_in=self.encoder.get_dim("x"), activate=True
        )
        self.parent_encoder = instantiate(
            self.cfg.word_encoder, n_in=self.encoder.get_dim("x"), activate=True
        )
        self.arc_encoder_w1 = nn.Parameter(
            torch.zeros(
                self.child_encoder.n_out,
                self.child_encoder.n_out,
                self.child_encoder.n_out,
            )
        )
        self.arc_encoder_w2 = nn.Parameter(
            torch.zeros(self.child_encoder.n_out, self.child_encoder.n_out)
        )
        self.arc_encoder_b = nn.Parameter(torch.zeros(self.child_encoder.n_out))

    @JointModelBase.add_impl_to_group(
        "lang_feat", "word+maxdep", "lang_feat_arc_mlp_init"
    )
    def lang_feat_max_tree(
        self,
        inputs: InputDict,
        language_encoded: TensorDict,
        lang_score: TensorDict,
        vp: VarPool,
    ):
        B, L, H = language_encoded["x"].shape
        L += 1

        mask = torch.cat([vp.mask.new_zeros(B, 1), vp.mask], dim=1)
        txt_mask = torch.cat([mask, mask], dim=1).refine_names("B", "Q")

        with torch.enable_grad():
            mdec = lang_score["merged_dec"].detach().requires_grad_()
            mattach = lang_score["merged_attach"].detach().requires_grad_()
            dist = DMV1o([mdec, mattach], vp.seq_len)
            arc_margin = torch.autograd.grad(dist.partition.sum(), mattach)[0].sum(-1)
            arc = dist.argmax.sum(-1).nonzero()
        predicted = vp.seq_len.new_zeros(vp.batch_size, vp.max_len + 1)
        predicted[arc[:, 0], arc[:, 2]] = arc[:, 1]

        if self.cfg.add_marginal:
            arc_margin = arc_margin.gather(-1, predicted.unsqueeze(-1)).squeeze(
                -1
            )  # batch x seq_len
        else:
            arc_margin = mask.to(arc_margin)
        txt_marginal = torch.cat([mask.to(arc_margin), arc_margin], dim=1).refine_names(
            "B", "Q"
        )

        x: Tensor = language_encoded["x"]
        root = (
            x.masked_fill(~vp.mask.unsqueeze(2), 0).sum(1) / vp.seq_len.unsqueeze(1)
        ).unsqueeze(1)
        x = torch.cat([root, x], dim=1)
        word_repr = self.word_encoder(x)

        child_repr = self.child_encoder(x)
        parent_repr = self.parent_encoder(
            x.gather(1, predicted.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        )
        arc_repr = (
            torch.einsum(
                "bcx,xhy,bcy->bch", child_repr, self.arc_encoder_w1, parent_repr
            )
            + torch.matmul(child_repr + parent_repr, self.arc_encoder_w2)
            + self.arc_encoder_b
        )
        # arc_repr = (child_repr + parent_repr) / 2
        txt = torch.cat([word_repr, arc_repr], dim=1).refine_names(
            "B", "Q", "X"
        )  # batch x (seq_len+seq_len) x hidden
        return txt, txt_mask, txt_marginal

    @JointModelBase.add_impl_to_group(
        "lang_feat", "word+alldep", "lang_feat_arc_mlp_init"
    )
    def lang_feat_all_arc(
        self,
        inputs: InputDict,
        language_encoded: TensorDict,
        lang_score: TensorDict,
        vp: VarPool,
    ):
        # NOT TESTED
        if not self.training:
            return self.lang_feat_max_tree(inputs, language_encoded, lang_score, vp)

        B, L, H = language_encoded["x"].shape
        L += 1

        mask = torch.cat([vp.mask.new_zeros(B, 1), vp.mask], dim=1)
        txt_mask = torch.cat(
            [mask, (mask.unsqueeze(1) * mask.unsqueeze(2)).view(B, -1)], dim=1
        )

        with torch.enable_grad():
            mdec = lang_score["merged_dec"].detach().requires_grad_()
            mattach = lang_score["merged_attach"].detach().requires_grad_()
            dist = DMV1o([mdec, mattach], vp.seq_len)
            arc_margin = torch.autograd.grad(dist.partition.sum(), mattach)[0].sum(-1)

        arc_margin = arc_margin.view(
            B, -1
        )  # head to dependent, 0 is root, NOTE diff from maxdep
        txt_marginal = torch.cat([torch.ones_like(arc_margin), arc_margin], dim=1)
        txt_marginal[:, 0] = 0  # mask root manually

        x: Tensor = language_encoded["x"]
        root = (
            x.masked_fill(~vp.mask.unsqueeze(2), 0).sum(1) / vp.seq_len.unsqueeze(1)
        ).unsqueeze(1)
        x = torch.cat([root, x], dim=1)
        word_repr = self.word_encoder(x)
        arc_repr = self.arc_encoder(x, x).view(B, L * L, -1)
        txt = torch.cat(
            [word_repr, arc_repr], dim=1
        )  # batch x (seq_len+seq_len) x hidden

        return txt, txt_mask, txt_marginal

    # feat_fuse ==============================================================

    def feat_fuse(self, encoded, vp):
        ...

    @JointModelBase.add_impl_to_group("feat_fuse", "none")
    def feat_fuse_none(self, encoded, vp):
        return encoded

    # noinspection PyAttributeOutsideInit
    def feat_fuse_attention_init(self):
        # noinspection PyTypeChecker
        cfg: DictConfig = self.cfg.feat_fuse_args
        self.attention = nn.MultiheadAttention(
            self.encoder.output_size,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            add_zero_attn=True,
            batch_first=True,
        )
        self.feat_layernorm = nn.LayerNorm(self.encoder.output_size)

    @JointModelBase.add_impl_to_group(
        "feat_fuse", "attention", "feat_fuse_attention_init"
    )
    def feat_fuse_attention(self, encoded, vp):
        """Args accepted:
        h_qk: required
        h_v: required
        sparsemax: optional, default to False
        to_align: optional, default to False. If true, the fused feature will affect the aligner.
        """

        vis_feat = [
            value
            for key, value in encoded.items()
            if key.startswith("vis")  # and key == "vis_box"
        ]
        if len(vis_feat) == 0:
            return encoded
        vis_feat = torch.cat(vis_feat, dim=1)

        # vis_feat = self.vis_mlp_pre_matching(vis_feat)

        if vis_feat.shape[1] == 0:
            return encoded
        if not self.cfg.feat_fuse_args.replace:
            encoded = {**encoded}
        # NOTE no fuse there, see _forward
        # if self.cfg.feat_fuse_args.aug_with_matching:
        #     # do layernorm later
        #     encoded["x"] = (
        #         encoded["x"] + self.attention(encoded["x"], vis_feat, vis_feat)[0]
        #     )
        # else:
        #     encoded["x"] = self.feat_layernorm(
        #         encoded["x"] + self.attention(encoded["x"], vis_feat, vis_feat)[0]
        #     )
        return encoded

    # gather_logit ===========================================================

    def gather_logit(self, inputs, vis, txt, vp):
        ...

    @JointModelBase.add_impl_to_group("gather_logit", "simple")
    def gather_logit_simple(self, inputs, vis, txt, vp):
        # gather loss for img-cap pair
        vis_feat, vis_mask, _ = vis
        txt_feat, txt_mask, txt_marginal = txt
        # [B1, K, dim] x [B2, querys, dim] => [B2, B1, querys, K]

        attmap: torch.Tensor = torch.einsum(
            "avd, bqd -> baqv", vis_feat.rename(None), txt_feat.rename(None)
        )
        attmap = attmap.refine_names("B", "A", "Q", "V")
        attmap.masked_fill_(~vis_mask.align_as(attmap), -INF)
        attmap.masked_fill_(~txt_mask.align_as(attmap), -INF)
        return attmap

    @JointModelBase.add_impl_to_group("gather_logit", "reduced")
    def gather_logit_reduced(self, inputs, vis, txt, vp):
        vis_feat, vis_mask, _ = vis
        txt_feat, txt_mask, txt_marginal = txt
        attmap = self.gather_logit_simple(inputs, vis, txt, vp)
        maxatt = attmap.max(dim=-1).values  # [B2, B1, querys]
        logit = torch.sum(
            maxatt * txt_marginal.unsqueeze(1), dim=-1
        ) / txt_marginal.sum(
            1, keepdim=True
        )  # [B2, B1]
        return logit

    # loss_grounding =========================================================

    def loss_grounding(self, inputs, vp):
        ...

    @JointModelBase.add_impl_to_group("loss_grounding", "factor|ce")
    def loss_grounding_factor_ce(self, inputs, vp):
        attmap: Tensor = inputs["match_logit"]
        txt_feat, txt_mask, txt_marginal = inputs["txt_packed"]
        vis_feat, vis_mask, vis_split = inputs["vis_packed"]

        if self.cfg.loss_grounding_args.use_pos_prior:
            names = attmap.names
            attmap = attmap.rename(None)
            arange = torch.arange(len(attmap), device=vis_mask.device)
            offset = 0
            for i, (name, width) in enumerate(zip(self.vis_factor_names, vis_split)):
                if name == "obj":
                    mask = (
                        vp.tag.unsqueeze(-1).eq(self.pos_for_obj).any(-1, keepdim=True)
                    )
                elif name == "rel":
                    mask = (
                        vp.tag.unsqueeze(-1).eq(self.pos_for_rel).any(-1, keepdim=True)
                    )
                elif name == "attr":
                    mask = (
                        vp.tag.unsqueeze(-1).eq(self.pos_for_attr).any(-1, keepdim=True)
                    )
                else:
                    offset += width
                    continue
                attmap[arange, arange, 1 : mask.shape[1] + 1, :offset] -= mask * 100
                attmap[arange, arange, 1 : mask.shape[1] + 1, offset + width :] -= (
                    mask * 100
                )
                offset += width
            attmap = attmap.refine_names(*names)

        logit = attmap.max("V").values

        _num = vp.num_token
        logit = logit.log_softmax("A")
        txt2vis = -(logit.rename(None).diagonal().T * txt_marginal).sum()
        loss = {"txt2vis": txt2vis / (txt2vis.detach() + 1e-6) * _num}

        if self.cfg.loss_grounding_args.vis2txt > 0:
            logit = attmap.max("Q").values
            logit = logit.log_softmax("B")
            vis2txt = -(logit.rename(None).diagonal().T * vis_mask).sum()
            loss["mt_vis2txt"] = (
                self.cfg.loss_grounding_args.vis2txt
                * vis2txt
                / (vis2txt.detach() + 1e-6)
                * _num
            )

        return sum(loss.values()), loss

    @JointModelBase.add_impl_to_group("loss_grounding", "cap_img|ce")
    def loss_grounding_cap_img_ll(self, inputs, vp):
        if not self.training:
            return 0, {}
        attmap = inputs["match_logit"]
        loss = self.criteria(attmap, torch.arange(vp.batch_size, device=attmap.device))
        return loss, {"mt": loss}

    # decode_match ===========================================================

    def decode_grounding(self, inputs, vp):
        ...

    @JointModelBase.add_impl_to_group("decode_grounding", "on_img")
    def decode_grounding_on_image(self, inputs, vp):
        return {
            "txt_to_img": inputs["match_logit"].argmax(1),
        }

    @JointModelBase.add_impl_to_group("decode_grounding", "on_factor")
    def decode_grounding_on_factor(self, inputs, vp):
        """
        The return value is a list L1 with batch size. The item is a list L2 represent a sentence.
        The item of L2 is a list L3 for each token. L3 contains id of box in descending order.
        """
        vis, vis_mask, vis_split = inputs["vis_packed"]
        match_logit: Tensor = inputs["match_logit"]  # [B2, B1, querys, K]
        factor2img = match_logit.max("V").values.max("A").indices

        match_logit = (
            match_logit.diagonal().refine_names("Q", "V", "B").align_to("B", "Q", "V")
        )  # [B, querys, K]
        match_logit = match_logit.rename(None)

        # >>>
        if self.cfg.decode_grounding_args.use_pos_prior:
            arange = torch.arange(len(match_logit), device=vis_mask.device)
            offset = 0
            with torch.no_grad():
                for i, (name, width) in enumerate(zip(self.vis_factor_names, vis_split)):
                    if name == "obj":
                        mask = (
                            vp.tag.unsqueeze(-1).eq(self.pos_for_obj).any(-1, keepdim=True)
                        )
                    elif name == "rel":
                        mask = (
                            vp.tag.unsqueeze(-1).eq(self.pos_for_rel).any(-1, keepdim=True)
                        )
                    elif name == "attr":
                        mask = (
                            vp.tag.unsqueeze(-1).eq(self.pos_for_attr).any(-1, keepdim=True)
                        )
                    else:
                        offset += width
                        continue
                    match_logit[arange, 1 : mask.shape[1] + 1, :offset] -= 1e10 * mask
                    match_logit[arange, 1 : mask.shape[1] + 1, offset + width :] -= (
                        1e10 * mask
                    )
                    offset += width

        if self.cfg.decode_grounding_args.use_heuristic:
            splitted_logit = torch.split(match_logit, vis_split, dim=-1)
            aligned_value: Tensor = match_logit.max(-1).values
            box_logit = splitted_logit[0]
            box_max_val, box_max_ind = box_logit.max(2)
            B, T, _ = splitted_logit[0].shape
            Barange = torch.arange(B, device=box_logit.device).unsqueeze(-1).expand(B, T)
            # Tarange = torch.arange(T, device=box_logit.device).unsqueeze(0).expand(B, T)
            if "rel" in self.vis_factor_names:
                rel_logit: Tensor = splitted_logit[self.vis_factor_names.index("rel")]
                with torch.no_grad():
                    allowed_box: Tensor = (box_max_val == aligned_value) & (
                        box_max_val > -1e5
                    )
                    allowed_box[:, vp.mask.shape[1] + 1:] = False  # arc do not contribute to allowed_box
                    allowed_mask = torch.zeros(
                        B, 1, vis_split[0], dtype=torch.bool, device=allowed_box.device
                    )
                    allowed_mask[Barange[allowed_box], 0, box_max_ind[allowed_box]] = True
                    allowed_mask = (
                        (allowed_mask.unsqueeze(-1) & allowed_mask.unsqueeze(-2))
                        .expand(B, T, -1, -1)
                        .view(B, T, -1)
                    )
                    rel_logit[~allowed_mask] -= 100
                    rel_logit = rel_logit.view(B, T, vis_split[0], vis_split[0])
                    rel_logit.diagonal(dim1=2, dim2=3).fill_(-1e10)
            if "attr" in self.vis_factor_names:
                attr_logit: Tensor = splitted_logit[self.vis_factor_names.index("attr")]
                with torch.no_grad():
                    allowed_box: Tensor = (box_max_val == aligned_value) & (
                        box_max_val > -1e5
                    )
                    allowed_mask = torch.zeros(
                        B, 1, vis_split[0], dtype=torch.bool, device=allowed_box.device
                    )
                    allowed_mask[Barange[allowed_box], 0, box_max_ind[allowed_box]] = True
                    attr_logit.masked_fill_(~allowed_mask, -1e10)
        # <<<

        match = match_logit.argsort(-1, descending=True)[..., :5].tolist()

        match_processed = []
        factor_start_point = [0] + list(accumulate(vis_split))
        vis_box_index = (
            vp.vis_box_index.tolist()
            if "vis_box_index" in vp
            else [list(range(200)) for _ in range(len(match_logit))]
        )
        for inst_match, l, box_index in zip(match, vp.seq_len_cpu, vis_box_index):
            inst_match_processed = []
            for candidates in inst_match:
                token_match_processed = []
                for idx in candidates:
                    factor_group = bisect_left(factor_start_point, idx)
                    if factor_start_point[factor_group] != idx:
                        factor_group -= 1
                    factor_name = self.vis_factor_names[factor_group]
                    idx -= factor_start_point[factor_group]
                    if factor_name == "rel":
                        idx = (
                            box_index[idx // vis_split[0]],
                            box_index[idx % vis_split[0]],
                        )
                    else:
                        idx = box_index[idx]
                    token_match_processed.append((factor_name, idx))
                inst_match_processed.append(token_match_processed)
            match_processed.append(inst_match_processed)

        return {
            "txt_to_factor": filter_list(
                match_processed, inputs["txt_packed"][1].tolist()
            ),
            "txt_to_img": filter_list(factor2img, inputs["txt_packed"][1].tolist()),
        }

    # write_render ===========================================================

    @JointModelBase.add_impl_to_group("format_factor_prediction", "on_img")
    def format_factor_prediction_on_img(self, factors, idx, length):
        if len(factors) > length:
            # assert len(factors) == 2 * length
            return "X\tX"
        return "X"  # placeholder

    @JointModelBase.add_impl_to_group("format_factor_prediction", "on_factor")
    def format_factor_prediction_on_boxrel(self, factors, idx, length):
        def _convert(x):
            t, x = x
            return f"{t} {x[0]}-{x[1]}" if isinstance(x, tuple) else f"{t} {x}"

        if len(factors) > length:  # word + dep
            assert len(factors) == 2 * length
            return "\t".join(
                [
                    "|".join(map(_convert, factors[idx])),
                    "|".join(map(_convert, factors[idx + length])),
                ]
            )
        return "|".join(map(_convert, factors[idx]))

    # api ====================================================================

    def _forward(self, inputs: InputDict, encoded: TensorDict, vp: VarPool):
        encoded = self.feat_fuse(encoded, vp)
        if (
            self.cfg.feat_fuse_mode != "none"
            and self.cfg.feat_fuse_args.aug_with_matching
        ):
            if encoded is not None and len(encoded) != 0:
                vis_encoded = {
                    k[4:]: v for k, v in encoded.items() if k.startswith("vis_")
                }
                vis = self.vis_feat(inputs, vis_encoded, vp, return_mid=True)
                txt = self.lang_feat_word_only(inputs, encoded, None, vp)
                attmap: torch.Tensor = torch.einsum(
                    "bvd, bqd -> bqv", vis[0].rename(None), txt[0].rename(None)[:, 1:]
                ).softmax(2)
                x = torch.einsum("bqv,bvh->bqh", attmap, vis[3])
                encoded["x"] = self.feat_layernorm(encoded["x"] + x)
        return self.dependency._forward(inputs, encoded, vp)

    def _vis_forward(
        self,
        inputs: InputDict,
        encoded: TensorDict,
        language_encoded: TensorDict,
        lang_score: TensorDict,
        vp: VarPool,
    ):
        if encoded is None or len(encoded) == 0:
            return {}

        vis = self.vis_feat(inputs, encoded, vp)
        txt = self.lang_feat(inputs, language_encoded, lang_score, vp)
        logit = self.gather_logit(inputs, vis, txt, vp)
        return {"match_logit": logit, "vis_packed": vis, "txt_packed": txt}

    def loss(
        self, x: TensorDict, gold: InputDict, vp: VarPool
    ) -> Tuple[Tensor, TensorDict]:
        # x: _forward(...) | _vis_forward(...), union of outputs
        alpha = self.cfg.grounding_interpolation
        dep_loss, dep_out = self.dependency.loss(x, gold, vp)

        if x.get("match_logit") is None or not self.training:
            return dep_loss, dep_out

        if alpha > 0 and vp.vis_available.sum() >= 2:
            mt_loss, mt_out = self.loss_grounding(x, vp)
        else:
            mt_loss, mt_out = 0, {}

        return alpha * mt_loss + (1 - alpha) * dep_loss, {
            **dep_out,
            **mt_out,
        }

    def decode(self, x: TensorDict, vp: VarPool) -> AnyDict:
        out = self.dependency.decode(x, vp)
        if x.get("match_logit") is None:
            return out
        return {**out, **self.decode_grounding({**x, **out}, vp)}

    def write_prediction(
        self, s: IOBase, predicts, dataset: DataSet, vocabs: Dict[str, Vocabulary]
    ) -> IOBase:
        tag_vocab = vocabs["tag"]
        for i, length in enumerate(dataset["seq_len"].content):
            word, tag, arc = (
                dataset[i]["raw_word"],
                dataset[i]["tag"],
                predicts["arc"][i],
            )
            factor = (
                predicts["txt_to_factor"][i]
                if "txt_to_factor" in predicts
                else [[]] * len(word)
            )
            for line_id, (word, tag, arc) in enumerate(zip(word, tag, arc), start=1):
                factor_token = self.format_factor_prediction(
                    factor, line_id - 1, length
                )
                line = "\t".join(
                    [str(line_id), word, tag_vocab.to_word(tag), str(arc), factor_token]
                )
                s.write(f"{line}\n")
            s.write("\n")
        return s

    def process_checkpoint(self, ckpt):
        state_dict = ckpt["state_dict"]
        if not any("dependency" in key for key in state_dict.keys()):
            new_state_dict = {}
            target_state_dict = self.state_dict().keys()
            for key, value in state_dict.items():
                if key not in target_state_dict and key.startswith("model."):
                    key = "model.dependency." + key[6:]
                elif key not in target_state_dict:
                    print(key)
                new_state_dict[key] = value
            ckpt["state_dict"] = new_state_dict
        ckpt_set = {
            k[6:] if k.startswith("model.") else k for k in ckpt["state_dict"].keys()
        }
        model_set = set(self.state_dict().keys())
        print(">>> model to ckpt")
        if len(model_set - ckpt_set):
            print(model_set - ckpt_set)
        print("=================")
        if len(ckpt_set - model_set):
            print(ckpt_set - model_set)
        print("<<< ckpt to model")
        return ckpt
