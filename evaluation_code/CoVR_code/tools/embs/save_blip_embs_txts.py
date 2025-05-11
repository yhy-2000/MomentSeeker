import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from src.data.utils import pre_caption
from src.model.blip.blip_cir import BLIPCir, blip_cir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(
        self,
        csv_path,
        column="txt2",
        max_words=30,
    ):
        self.df = pd.read_csv(csv_path)
        self.texts = list(set(self.df[column].unique().tolist()))
        self.texts.sort()
        self.max_words = max_words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        txt = self.texts[index]
        txt = pre_caption(txt, self.max_words)

        return txt
    
class MyTextDataset(Dataset):
    def __init__(
        self,
        csv_path,
        column="txt2",
        max_words=30,
    ):
        with open(csv_path, "r") as f:
            data = json.load(f)
        self.texts = []
        self.img_pth = []

        for idx, item in enumerate(data):
            self.texts.append({"query": item['qry_text'], "id": idx})
            self.img_pth.append(item['qry_video_path'])
        self.max_words = max_words
        print(f"number of texts: {len(self.texts)}")

        self.image_size = 384

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        txt = self.texts[index]['query']
        txt = pre_caption(txt, self.max_words)
        img_name = self.img_pth[index].split("/")[-1]
        img_path = f"/share/liangzy/CoVR_code/qry_imgsv2/{img_name}.jpg"
        img = Image.open(img_path).convert('RGB')
        img = torch.tensor(self.transform(img))
        return txt, img


def get_blip_config(model="modify"):
    config = dict()

    config["pretrained"] = (
        "/share/liangzy/CoVR_code/outputs/webvid-covr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-1e-05/good/ckpt_4.ckpt"
    )
    config["vit"] = "large"
    config["batch_size_train"] = 16
    config["batch_size_test"] = 32
    config["vit_grad_ckpt"] = True
    config["vit_ckpt_layer"] = 12
    config["init_lr"] = 5e-6

    config["image_size"] = 384
    config["queue_size"] = 57600
    config["alpha"] = 0.4
    config["k_test"] = 256
    config["negative_all_rank"] = True

    return config


@torch.no_grad()
def main(args):
    dataset = MyTextDataset(args.data_csv, column=args.column)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print("Creating model")
    config = get_blip_config(args.model_type)
    model = BLIPCir(
        loss=":)",
        med_config="configs/med_config.json",
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
    )
    model = blip_cir(model, config["pretrained"])

    model = model.to(device)
    model.eval()

    text_feats = []
    for txts, imgs in tqdm(loader):
        print(txts)
        txts = model.tokenizer(
            txts,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        imgs = imgs.to(device)
        # text_output = model.text_encoder(
        #     txts.input_ids,
        #     attention_mask=txts.attention_mask,
        #     return_dict=True,
        #     mode="text",
        # )
        ref_img_embs = model.visual_encoder(imgs)
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(
            device
        )
        text_output = model.text_encoder(
            txts.input_ids,
            attention_mask=txts.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        text_feat = F.normalize(
            model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        ).cpu()
        text_feats.append(text_feat)

    text_feats = torch.cat(text_feats, dim=0)
    save_obj = {
        "texts": dataset.texts,
        "feats": text_feats,
    }
    torch.save(save_obj, args.save_dir / f"{args.column}_{args.data_csv.stem}.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_csv", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--model_type", type=str, default="modify", choices=["base", "large", "modify"]
    )
    parser.add_argument("--column", type=str, default="txt2")
    args = parser.parse_args()

    args.save_dir.mkdir(exist_ok=True)

    main(args)
