
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
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import numpy as np
import torch
from transformers import AutoModel
import easydict
import sys
sys.path.append('./VLM2Vec')
from src.utils import load_processor
import json


from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor

from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred



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
        index_video_clips.extend([clip['frame_path'] for clip in dic['candidate_video_list']])

    index_video_clips = list(set(index_video_clips))
    eval_dataset.index_examples = index_video_clips

    def process_query_example(query):
        qtext = query["qry_text"]

        if query["qry_img_path"]!='': # ti2v
            ima = query["qry_img_path"]
        elif query["qry_video_path"]!='': # tv2v
            ima = get_medium_frame_path(query["qry_video_path"])
        else: # t2v
            ima = None

        gt_clips = [query['candidate_video_list'][i]['frame_path'] for i in query['gt_indices']]
        assert len(gt_clips) > 0, f"gt_clips is empty: {qtext} {file}"
        return QueryExample(qid=query["qry_text"], qtokens=qtext, qimage=ima, target_iid=gt_clips, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing query examples...")
        query_examples = list(executor.map(process_query_example, queries))
        eval_dataset.query_examples = query_examples

    return eval_dataset



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    model_args = ModelArguments(
        model_name="/share/huaying/pretrained_model/VLM2Vec-LLaVa-Next",
        pooling='last',
        normalize=True,
        model_backbone='llava_next'
    )
    processor = load_processor(model_args)

    model = MMEBModel.load(model_args)
    model.eval()
    model = model.cuda().to(dtype=torch.bfloat16)


    output_dir = './languagebind/benchmark/test_grounding/result/vlm2vec/'
    os.makedirs(output_dir,exist_ok=True)

    benchmark_dir = './languagebind/benchmark/test_grounding/'
    for file in os.listdir(benchmark_dir):
        try:
            if not file.endswith('.json') or 'msrvtt' in file:
                continue
            
            output_file = os.path.join(output_dir, f"{file.split('.')[0]}_results.json")
            # if os.path.exists(output_file):
            #     continue

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
                # inference index:
                index_embeddings = []
                print("Inference index...")
                for i in tqdm(range(0, len(eval_dataset.index_examples), args.batch_size)):
                    batch_video = eval_dataset.index_examples[i : i + args.batch_size]

                    inputs = processor(text=[f"Represent the given image in one word: " for _ in batch_video],
                        images=[Image.open(frame_path) for frame_path in batch_video],
                        return_tensors="pt")
                    
                    for key, value in inputs.items():
                        inputs[key] = torch.tensor(value).to('cuda')
                    
                    # inputs = {key: value.to('cuda') for key, value in inputs.items()}
                    video_embed = model(qry=inputs)["qry_reps"].float().detach().cpu().numpy()
                    
                    index_embeddings.append(video_embed)
                index_embeddings = np.concatenate(index_embeddings, axis=0)

                pickle.dump(index_embeddings,open(f'{output_dir}/{file.split(".")[0]}_index.pkl','wb'))


            total_query_embeds = []
            print("Inference queries...")
            for i in tqdm(range(0, len(eval_dataset.query_examples))):
                batch = eval_dataset.query_examples[i : i + args.batch_size]
                if batch[0].qimage is not None:                
                    inputs = processor(text=[q.qtokens for q in batch],
                        images=[Image.open(q.qimage) for q in batch],
                        return_tensors="pt")
                    inputs = {key: value.to('cuda') for key, value in inputs.items()}
                    query_embeds = model(qry=inputs)["qry_reps"].float().detach().cpu().numpy()
                else:
                    inputs = processor(text=[q.qtokens for q in batch],
                                    images=None,
                                    return_tensors="pt")
                    inputs = {key: value.to('cuda') for key, value in inputs.items()}
                    query_embeds = model(qry=inputs)["qry_reps"].float().detach().cpu().numpy()


                similarity_scores = np.dot(query_embeds, index_embeddings.T)

                # get top 50 by similarity (by default)
                top_k_indices = np.argsort(similarity_scores, axis=1)[:, ::-1]
                
                top_k_iids = []
                for j in range(len(batch)):
                    cur_top_k = top_k_indices[j]
                    gt_video_name = [file.split('/')[-2] for file in batch[j].target_iid]
                    domain_top_k_iids = []
                    domain_top_k = []

                    tmp = []
                    for k in cur_top_k:
                        video_path = eval_dataset.index_examples[k]
                        video_name = video_path.split('/')[-2]
                        tmp.append(video_name)

                        if video_name in gt_video_name:
                            domain_top_k_iids.append(video_path)
                            domain_top_k.append(k)

                    top_k_iids.append(domain_top_k_iids[:50])
                    domain_top_k = domain_top_k[:50]

                    if domain_top_k_iids[:50]==[]:
                        print(cur_top_k)
                        print(batch[j].target_iid)
                        print(tmp)
                    
                    eval_dataset.query_examples[i + j].retrieved_iids = domain_top_k_iids[:50]
                    eval_dataset.query_examples[i + j].retrieved_scores = similarity_scores[j, domain_top_k].tolist()
                            
            
            eval_dataset.evaluate_recall(output_dir)
            eval_dataset.evaluate_map(output_dir)
            eval_dataset.write_to_file(output_dir)



        except Exception as e:
            print(e)





