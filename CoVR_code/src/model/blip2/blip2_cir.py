"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather


class BLIP2Cir(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="clip_L",
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        temperature=1,
        si_ti_weight=1,
        si_tc_weight=0,
    ):
        super().__init__()

        self.loss = loss

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = temperature

        self.max_txt_len = max_txt_len

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        for p in self.ln_vision.parameters():
            p.requires_grad = False

        for p in self.Qformer.cls.parameters():
            p.requires_grad = False

        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

    def forward(self, batch, fabric):
        ref_img = batch["ref_img"]
        tar_img_feat = batch["tar_img_feat"]
        caption = batch["edit"]

        ref_img.half()

        device = ref_img.device

        # Encode the target image
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = concat_all_gather(tar_img_feat, fabric)

        # Text
        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        if self.train_vit:
            ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
        else:
            with torch.no_grad():
                ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

        # Encode the reference image
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        ###============== Image-text Matching ===================###
        query_tokens = self.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        output = self.Qformer.bert(
            text_tokens.input_ids,  # [bs, 32]
            query_embeds=query_tokens,  # [bs, 32, 768]
            attention_mask=attention_mask,  # [bs, 64]
            encoder_hidden_states=ref_img_embs,  # [bs, 677, 1408]
            encoder_attention_mask=ref_img_atts,  # [bs, 677]
            return_dict=True,
        )

        vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
        query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
        query_si_feat = all_gather_with_grad(query_si_feat, fabric)

        # mean over all query tokens
        query_si_feat = query_si_feat.mean(dim=1)
        tar_img_feat = tar_img_feat.mean(dim=1)

        # s=source, t=target, i=image, c=caption, w=weight
        loss = 0
        if self.si_ti_weight > 0:
            si_ti_loss = self.loss(query_si_feat, tar_img_feat, self.temp)
            loss += si_ti_loss * self.si_ti_weight

        if self.si_tc_weight > 0:
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]

            tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)

            si_tc_loss = self.loss(query_si_feat, tar_txt_feat, self.temp)
            loss += si_tc_loss * self.si_tc_weight

        return loss


def blip2_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model
