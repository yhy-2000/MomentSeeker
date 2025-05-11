# source activate /share/huaying/envs/covr

import json
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import sys
sys.path.append('./magiclens/')
from data_utils import *
import multiprocessing as mp
import os
import json
import numpy as np
import pickle
from typing import Dict, List
from argparse import ArgumentParser
from tqdm import tqdm

import cv2
import numpy as np

import torch
from transformers import AutoModel

sys.path.append('./CoVR_code/')

from src.data.embs import MyVideoDataset
from src.model.blip.blip_embs import blip_embs 
from src.data.utils import pre_caption
from src.model.blip.blip_cir import BLIPCir, blip_cir

from torchvision.transforms.functional import InterpolationMode
import cv2


transform = transforms.Compose(
            [
                transforms.Resize(
                    (384, 384),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

def sample_frames(vlen, frames_per_video=15):
    acc_samples = min(vlen, frames_per_video)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))

    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs


def get_video_frames(video_pth, frames_video=15, image_size=384):
    video_pth = str(video_pth)
    cap = cv2.VideoCapture(video_pth)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(video_pth, total_frames, frames_video)
    frame_idxs = sample_frames(total_frames, frames_video)

    frames = []
    f_idxs = []
    for frame_idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"Video {video_pth} is corrupted")
            frames = [
                Image.fromarray(np.zeros((image_size, image_size, 3)).astype(np.uint8))
            ] * frames_video
            f_idxs = [-1] * frames_video
            return frames, f_idxs

        frames.append(Image.fromarray(frame))
        f_idxs.append(frame_idx)

    n_frames = len(frames)
    if n_frames < frames_video:
        frames += [
            Image.fromarray(np.zeros((image_size, image_size, 3)).astype(np.uint8))
        ] * (frames_video - n_frames)
    f_idxs += [-1] * (frames_video - len(f_idxs))

    for i,frame in enumerate(frames):
        frames[i] = transform(frames[i])

    frames = torch.stack(frames)
    f_idxs = torch.tensor(f_idxs)
    return frames



def get_blip_config(model="modify"):
    config = dict()
    config["pretrained"] = (
        "/share/liangzy/CoVR_code/outputs/webvid-covr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-1e-05/good/ckpt_4.ckpt"
    )
    config["vit"] = "large"
    config["batch_size_train"] = 48
    config["batch_size_test"] = 96
    config["vit_grad_ckpt"] = True
    config["vit_ckpt_layer"] = 12
    config["init_lr"] = 5e-6

    config["image_size"] = 384
    config["queue_size"] = 57600
    config["alpha"] = 0.4
    config["k_test"] = 256
    config["negative_all_rank"] = True
    return config


@dataclass
class QueryExample:
    qid: str
    qtokens: np.ndarray
    qimage: np.ndarray
    target_iid: Union[int, str, List[int], List[str], None] # can be int or 
    retrieved_iids: List[Union[int, str]] # ranked by score, can be str (cirr) or int (circo)
    retrieved_scores: List[float] # ranked by order


@dataclass
class IndexExample:
    iid: Union[int, str]
    iimage: np.ndarray
    itokens: np.ndarray


@dataclass
class Dataset:
    name: str
    query_examples: List[QueryExample] = field(default_factory=list)
    k_range: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    index_examples: List[IndexExample] = field(default_factory=list)

    def evaluate_recall(self,output_dir):
        ret_dict = {k: [] for k in self.k_range}

        for q_example in self.query_examples:
            assert len(q_example.retrieved_iids) > 0, "retrieved_iids is empty"
            for k in self.k_range:
                recalled = False
                if isinstance(q_example.target_iid, list):
                    for one_target_iid in q_example.target_iid:
                        if one_target_iid in q_example.retrieved_iids[:k]:
                            recalled = True
                elif isinstance(q_example.target_iid, int) or isinstance(q_example.target_iid, str):
                    if q_example.target_iid in q_example.retrieved_iids[:k]:
                        recalled = True
                else:
                    raise ValueError(f"target_iid is of type {type(q_example.target_iid)}")

                if recalled:
                    ret_dict[k].append(1)
                else:
                    ret_dict[k].append(0)
        total_ex = len(self.query_examples)
        ret_dict = {k: round((sum(v) / total_ex) * 100, 2) for k, v in ret_dict.items()}
        print("Recalls: ", ret_dict)

        output_file = os.path.join(output_dir, f"{self.name}_results.log")
        with open(output_file, "w") as f:
            json.dump(ret_dict, f, indent=4)

        return ret_dict


    def evaluate_map(self, output_dir):
        """
        计算 mAP@5：
        - 每个查询只考虑前 5 个检索结果；
        - 相关度用当前视频片段对应的时间区间与所有 ground‑truth
            （即所有 ground‑truth 区间拼接成一个整体区间）之间的 IoU 表示；
        - 不论候选视频是否为 ground‑truth，都计算 IoU 得分；
        - 采用加权平均的方法计算 AP（假设整体 ground‑truth视为1个目标）。
        """
        aps = []  # 存储每个查询的 AP

        def parse_interval(iid):
            """
            从 iid 字符串中解析时间区间，假设 iid 格式为：
            "/share/huaying/long_video/.../movie101_16/0.00_3.00.mp4"
            """
            base = os.path.basename(iid)           # 例如 "0.00_3.00.mp4"
            base_no_ext, _ = os.path.splitext(base)  # 得到 "0.00_3.00"
            parts = base_no_ext.split('_')
            if len(parts) >= 2:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    return (start, end)
                except ValueError:
                    return (0.0, 0.0)
            else:
                return (0.0, 0.0)

        def compute_iou(interval1, interval2):
            """
            计算两个时间区间的 IoU（交并比）
            """
            start1, end1 = interval1
            start2, end2 = interval2
            inter = max(0.0, min(end1, end2) - max(start1, start2))
            union = max(end1, end2) - min(start1, start2)
            return inter / union if union > 0 else 0.0

        for q_example in self.query_examples:
            # 解析 ground‑truth 的 iid，并提取所有对应的时间区间
            if isinstance(q_example.target_iid, list):
                gt_iids = q_example.target_iid
            else:
                gt_iids = [q_example.target_iid]
            gt_intervals = [parse_interval(gt) for gt in gt_iids]

            if not gt_intervals:
                aps.append(0.0)
                continue  # 如果没有 ground‑truth，则 AP 定义为 0

            # 将所有 ground‑truth 区间拼接成一个整体区间：
            # 取所有 ground‑truth 的最小起始时间和最大结束时间
            union_start = min(interval[0] for interval in gt_intervals)
            union_end = max(interval[1] for interval in gt_intervals)
            union_interval = (union_start, union_end)

            # 只考虑前 5 个检索结果
            retrieved = q_example.retrieved_iids[:5]

            cumulative_relevance = 0.0  # 累计 IoU 得分
            sum_precision = 0.0         # 加权累计精度

            # 遍历前 5 个检索结果（位置从 1 开始计数）
            for pos, iid in enumerate(retrieved, start=1):
                r_interval = parse_interval(iid)
                # 直接计算当前视频片段与整体 ground‑truth 区间之间的 IoU
                iou = compute_iou(r_interval, union_interval)
                cumulative_relevance += iou
                precision_at_pos = cumulative_relevance / pos
                # 加权累加：当前位置的精度乘以当前 IoU 得分
                sum_precision += precision_at_pos * iou

            # 此处将整体 ground‑truth视为1个目标
            ap = sum_precision  # 或者 ap = sum_precision / 1
            aps.append(ap)

        # 计算 mAP@5（取所有查询 AP 的均值），并转换为百分比表示
        mAP = sum(aps) / len(aps) if aps else 0.0
        mAP_percentage = round(mAP * 100, 2)
        map_result = {'mAP@5': mAP_percentage}

        print("mAP@5: ", map_result)

        # 写入日志文件
        output_file = os.path.join(output_dir, f"{self.name}_results.log")
        with open(output_file, "a+") as f:
            f.seek(0)  # 定位到文件开始位置，读取现有内容
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
            existing_data.update(map_result)  # 合并新的结果
            f.seek(0)
            f.truncate()
            json.dump(existing_data, f, indent=4)

        return map_result

    def write_to_file(self, output_dir: str):
        os.makedirs(output_dir,exist_ok=True)

        dict_to_write = dict()
        for q_example in self.query_examples:
            dict_to_write[q_example.qid] = {
                'gt': q_example.target_iid,
                'pred': q_example.retrieved_iids[:50]
            }
        output_file = os.path.join(output_dir, f"{self.name}_results.json")
        with open(output_file, "w") as f:
            json.dump(dict_to_write, f, indent=4)
        print("Results are written to file", output_file)



def get_medium_frame_path(video_path: str):
    if os.path.exists(video_path.replace('.mp4','_mid_frame.png')):
        return video_path.replace('.mp4','_mid_frame.png')
    else:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count//2)
        ret, frame = cap.read()
        cv2.imwrite(video_path.replace('.mp4','_mid_frame.png'), frame)
        return video_path.replace('.mp4','_mid_frame.png')


def build_dataset(file: str) -> Dataset:
    eval_dataset = Dataset(file.split('/')[-1].split('.')[0])
    queries = json.load(open(file))

    index_video_clips = []
    for dic in queries:
        index_video_clips.extend([clip['output_path'] for clip in dic['candidate_video_list']])

    index_video_clips = list(set(index_video_clips))
    eval_dataset.index_examples = sorted(index_video_clips)

    def process_query_example(query):
        qtext = query["qry_text"]

        if query["qry_img_path"]!='': # ti2v
            ima = query["qry_img_path"]
        elif query["qry_video_path"]!='': # tv2v
            ima = query["qry_video_path"]
        else: # t2v
            ima = None

        gt_clips = [query['candidate_video_list'][i]['output_path'] for i in query['gt_indices']]
        return QueryExample(qid=query["qry_text"], qtokens=qtext, qimage=ima, target_iid=gt_clips, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing query examples...")
        query_examples = list(executor.map(process_query_example, queries))
        eval_dataset.query_examples = query_examples

    return eval_dataset



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.2f}M")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    config = get_blip_config()
    single_modality_model = blip_embs(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        negative_all_rank=config["negative_all_rank"],
    )

    single_modality_model = single_modality_model.to(device)
    single_modality_model.eval()

    config = get_blip_config()
    multi_modality_model = BLIPCir(
        loss=":)",
        med_config="./CoVR_code/configs/med_config.json",
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
    )
    multi_modality_model = blip_cir(multi_modality_model, config["pretrained"])

    multi_modality_model = multi_modality_model.cuda()
    multi_modality_model.eval()
    num_index_gpus = torch.cuda.device_count()

    output_dir = './languagebind/benchmark/test_grounding/result/covr/'
    os.makedirs(output_dir,exist_ok=True)


    benchmark_dir = './languagebind/benchmark/test_grounding/'
    for file in os.listdir(benchmark_dir):
        try:
            if not file.endswith('.json') or 'msrvtt' in file or 'didemo' in file or 'msvd' in file or 'vatex' in file:
                continue
            
            # if 'tv2v' not in file and 'ti2v' not in file:
            #     continue
            
            output_file = os.path.join(output_dir, f"{file.split('.')[0]}_results.json")
            if os.path.exists(os.path.join(output_dir, f"{file.split('.')[0]}_results.lock")):
                continue

            with open(os.path.join(output_dir, f"{file.split('.')[0]}_results.lock"),'w') as fw:
                print('lock',file=fw)
            
            print('processing',file)
                    
            eval_dataset = build_dataset(os.path.join(benchmark_dir,file))             

            ok = 0
            index_path = os.path.join(output_dir, f"{file.split('.')[0]}_index.pkl")
            index_examples_path = os.path.join(output_dir, f"{file.split('.')[0]}_index_examples.pkl")

            if os.path.exists(index_path) and os.path.exists(index_examples_path):
                index_examples = pickle.load(open(index_examples_path,'rb'))
                if index_examples == eval_dataset.index_examples:
                    ok=1
                    with open(index_path, 'rb') as f:
                        index_embeddings = pickle.load(f)
                    assert len(index_embeddings) == len(index_examples), f'{len(index_embeddings)} != {len(index_examples)}'


            if not ok:
                index_embeddings = []
                print('#'*100)
                print("Inference index...")
                print('#'*100)
                for i in tqdm(range(0, len(eval_dataset.index_examples), args.batch_size)):
                    batch_video = eval_dataset.index_examples[i : i + args.batch_size]
                    frames = [get_video_frames(video_path) for video_path in batch_video]
                    frames = torch.stack(frames).to(device)
                    bs, nf, c, h, w = frames.shape
                    frames = frames.view(bs * nf, c, h, w)
                    video_embed = single_modality_model.visual_encoder(frames)
                    video_embed = F.normalize(single_modality_model.vision_proj(video_embed[:, 0, :]), dim=-1).cpu()
                    video_embed = video_embed.view(bs, nf, -1)

                    index_embeddings.append(video_embed.detach().cpu())
                index_embeddings = torch.stack(index_embeddings, axis=0).squeeze()
                pickle.dump(index_embeddings,open(f'{output_dir}/{file.split(".")[0]}_index.pkl','wb'))
                pickle.dump(eval_dataset.index_examples,open(f'{output_dir}/{file.split(".")[0]}_index_examples.pkl','wb'))

            else:
                print('#'*100)
                print('loading index from:',index_path)
                print('#'*100)


            total_query_embeds = []
            print("Inference queries...")
            for i in tqdm(range(0, len(eval_dataset.query_examples), args.batch_size)):
                batch = eval_dataset.query_examples[i : i + args.batch_size]
                if batch[0].qimage is not None:
                    text_inputs = multi_modality_model.tokenizer(
                        [q.qtokens for q in batch],
                        padding="max_length",
                        truncation=True,
                        max_length=50,
                        return_tensors="pt",
                    ).to(device)

                    imgs = get_video_frames([q.qimage for q in batch]).to(device)
                    ref_img_embs = multi_modality_model.visual_encoder(imgs)
                    ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

                    

                    text_output = multi_modality_model.text_encoder(
                        text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask,
                        encoder_hidden_states=ref_img_embs,
                        encoder_attention_mask=ref_img_atts,
                        return_dict=True,
                    )
                    
                    query_embeds = F.normalize(
                        multi_modality_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                    ).detach().cpu()
                else:
                    text_inputs = multi_modality_model.tokenizer(
                        [q.qtokens for q in batch],
                        padding="max_length",
                        truncation=True,
                        max_length=50,
                        return_tensors="pt",
                    ).to(device)
                    text_output = multi_modality_model.text_encoder(
                        text_inputs.input_ids,
                        attention_mask=text_inputs.attention_mask,
                        return_dict=True,
                    )
                    query_embeds = F.normalize(
                        multi_modality_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
                    ).detach().cpu()

                similarity_scores = torch.einsum('ad,bfd->abf', query_embeds, index_embeddings.squeeze())
                similarity_scores = similarity_scores.sum(dim=-1).numpy()

                # get top 50 by similarity (by default)
                top_k_indices = np.argsort(similarity_scores, axis=1)[:, ::-1]
                
                top_k_iids = []
                for j in range(len(batch)):
                    cur_top_k = top_k_indices[j]
                    gt_video_name = [file.split('/')[-2] for file in batch[j].target_iid]
                    domain_top_k_iids = []
                    domain_top_k = []

                    for k in cur_top_k:
                        video_path = eval_dataset.index_examples[k]
                        video_name = video_path.split('/')[-2]
                        if video_name in gt_video_name:
                            domain_top_k_iids.append(video_path)
                            domain_top_k.append(k)

                    top_k_iids.append(domain_top_k_iids[:50])
                    domain_top_k = domain_top_k[:50]

                    eval_dataset.query_examples[i + j].retrieved_iids = domain_top_k_iids[:50]
                    eval_dataset.query_examples[i + j].retrieved_scores = similarity_scores[j, domain_top_k].tolist()
                            
    

            eval_dataset.evaluate_recall(output_dir)
            eval_dataset.evaluate_map(output_dir)
            eval_dataset.write_to_file(output_dir)
        except Exception as e:
            print(e)
