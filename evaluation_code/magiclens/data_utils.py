# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=redefined-outer-name,missing-module-docstring,g-importing-member,missing-function-docstring,g-bare-generic,g-doc-args,missing-class-docstring
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from typing import Any, List, Union
from tqdm import tqdm

import numpy as np
from PIL import Image


@dataclass
class QueryExample:
    qid: str
    qtokens: np.ndarray
    qimage: np.ndarray
    target_iid: Union[int, str, List[int], List[str], None] # can be int or 
    retrieved_iids: List[Union[int, str]] # ranked by score, can be str (cirr) or int (circo)
    retrieved_scores: List[float] # ranked by order
    candidate_video_list: Union[int, str, List[int], List[str], None] # can be int or 


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



# @dataclass
# class Dataset:
#     name: str
#     query_examples: List[QueryExample] = field(default_factory=list)
#     k_range: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
#     # write_to_file_header: Dict[str, Any] = field(default_factory=dict)
#     index_examples: List[IndexExample] = field(default_factory=list)


#     def evaluate_recall(self,output_dir):
#         ret_dict = {k: [] for k in self.k_range}

#         for q_example in self.query_examples:
#             assert len(q_example.retrieved_iids) > 0, "retrieved_iids is empty"
#             for k in self.k_range:
#                 recalled = False
#                 if isinstance(q_example.target_iid, list):
#                     for one_target_iid in q_example.target_iid:
#                         if one_target_iid in q_example.retrieved_iids[:k]:
#                             recalled = True
#                 elif isinstance(q_example.target_iid, int) or isinstance(q_example.target_iid, str):
#                     if q_example.target_iid in q_example.retrieved_iids[:k]:
#                         recalled = True
#                 else:
#                     raise ValueError(f"target_iid is of type {type(q_example.target_iid)}")

#                 if recalled:
#                     ret_dict[k].append(1)
#                 else:
#                     ret_dict[k].append(0)
#         # calculation
#         total_ex = len(self.query_examples)
#         ret_dict = {k: (sum(v) / total_ex) * 100 for k, v in ret_dict.items()}
#         print("Recalls: ", ret_dict)

#         output_file = os.path.join(output_dir, f"{self.name}_results.log")
#         with open(output_file, "a+") as f:
#             json.dump(ret_dict, f, indent=4)

#         return ret_dict

#     def evaluate_map(self, output_dir):
#         """
#         计算 mAP@5：
#         - 每个查询只考虑前 5 个检索结果；
#         - 相关度用当前视频片段对应的时间区间与所有 ground‑truth
#             （即所有 ground‑truth 区间拼接成一个整体区间）之间的 IoU 表示；
#         - 不论候选视频是否为 ground‑truth，都计算 IoU 得分；
#         - 采用加权平均的方法计算 AP（假设整体 ground‑truth视为1个目标）。
#         """
#         aps = []  # 存储每个查询的 AP

#         def parse_interval(iid):
#             """
#             从 iid 字符串中解析时间区间，假设 iid 格式为：
#             "/share/huaying/long_video/.../movie101_16/0.00_3.00.mp4"
#             """
#             base = os.path.basename(iid)           # 例如 "0.00_3.00.mp4"
#             base_no_ext, _ = os.path.splitext(base)  # 得到 "0.00_3.00"
#             parts = base_no_ext.split('_')
#             if len(parts) >= 2:
#                 try:
#                     start = float(parts[0])
#                     end = float(parts[1])
#                     return (start, end)
#                 except ValueError:
#                     return (0.0, 0.0)
#             else:
#                 return (0.0, 0.0)

#         def compute_iou(interval1, interval2):
#             """
#             计算两个时间区间的 IoU（交并比）
#             """
#             start1, end1 = interval1
#             start2, end2 = interval2
#             inter = max(0.0, min(end1, end2) - max(start1, start2))
#             union = max(end1, end2) - min(start1, start2)
#             return inter / union if union > 0 else 0.0

#         for q_example in self.query_examples:
#             # 解析 ground‑truth 的 iid，并提取所有对应的时间区间
#             if isinstance(q_example.target_iid, list):
#                 gt_iids = q_example.target_iid
#             else:
#                 gt_iids = [q_example.target_iid]
#             gt_intervals = [parse_interval(gt) for gt in gt_iids]

#             if not gt_intervals:
#                 aps.append(0.0)
#                 continue  # 如果没有 ground‑truth，则 AP 定义为 0

#             # 将所有 ground‑truth 区间拼接成一个整体区间：
#             # 取所有 ground‑truth 的最小起始时间和最大结束时间
#             union_start = min(interval[0] for interval in gt_intervals)
#             union_end = max(interval[1] for interval in gt_intervals)
#             union_interval = (union_start, union_end)

#             # 只考虑前 5 个检索结果
#             retrieved = q_example.retrieved_iids[:5]

#             cumulative_relevance = 0.0  # 累计 IoU 得分
#             sum_precision = 0.0         # 加权累计精度

#             # 遍历前 5 个检索结果（位置从 1 开始计数）
#             for pos, iid in enumerate(retrieved, start=1):
#                 r_interval = parse_interval(iid)
#                 # 直接计算当前视频片段与整体 ground‑truth 区间之间的 IoU
#                 iou = compute_iou(r_interval, union_interval)
#                 cumulative_relevance += iou
#                 precision_at_pos = cumulative_relevance / pos
#                 # 加权累加：当前位置的精度乘以当前 IoU 得分
#                 sum_precision += precision_at_pos * iou

#             # 此处将整体 ground‑truth视为1个目标
#             ap = sum_precision  # 或者 ap = sum_precision / 1
#             aps.append(ap)

#         # 计算 mAP@5（取所有查询 AP 的均值），并转换为百分比表示
#         mAP = sum(aps) / len(aps) if aps else 0.0
#         mAP_percentage = round(mAP * 100, 2)
#         map_result = {'mAP@5': mAP_percentage}

#         print("mAP@5: ", map_result)

#         # 写入日志文件
#         output_file = os.path.join(output_dir, f"{self.name}_results.log")
#         with open(output_file, "a+") as f:
#             f.seek(0)  # 定位到文件开始位置，读取现有内容
#             try:
#                 existing_data = json.load(f)
#             except json.JSONDecodeError:
#                 existing_data = {}
#             existing_data.update(map_result)  # 合并新的结果
#             f.seek(0)
#             f.truncate()
#             json.dump(existing_data, f, indent=4)

#         return map_result



#     def write_to_file(self, output_dir: str):
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         dict_to_write = dict()
#         for q_example in self.query_examples:
#             dict_to_write[q_example.qid] = q_example.retrieved_iids[:50]
#         output_file = os.path.join(output_dir, f"{self.name}_results.json")
#         with open(output_file, "w") as f:
#             json.dump(dict_to_write, f, indent=4)
#         print("Results are written to file", output_file)




def build_fiq_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    subtask = dataset_name.split("-")[1]
    queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.val.json"))
    index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.val.json"))
    index_image_folder = "./data/fiq/images"

    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".png")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".png")
        ima = process_img(qimage_path, 224)
        qtokens = tokenizer(qtext)
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset


def build_circo_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    queries = json.load(open("./data/circo/annotations/test.json"))
    coco_info = json.load(open("./data/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json"))
    index_img_ids = [img_info['id'] for img_info in coco_info['images']]
    index_image_folder = "./data/circo/COCO2017_unlabeled/unlabeled2017"

    def image_id2name(image_id):
        return str(image_id).zfill(12) + '.jpg'

    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, image_id2name(index_img_id))
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['id']
        qtext = f"find {query['shared_concept']} but {query['relative_caption']}"
        qimage_path = os.path.join(index_image_folder, image_id2name(query['reference_img_id']))
        ima = process_img(qimage_path, 224)
        qtokens = np.array(tokenizer(qtext))
        # circo test does not provide target id.
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset


# TODO test
def build_dtin_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    all_domains = ['cartoon', 'origami', 'toy', 'sculpture']   # only evaluate on one domain per run
    target_domain = dataset_name.split("-")[1]
    query_entries = open("./data/dtin/imgnet_real_query.txt").readlines()
    index_entries = open("./data/dtin/imgnet_targets.txt").readlines()

    # debug
    # query_entries = query_entries[:10]
    # index_entries = index_entries[:10]
    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    query_text_template = "find this object in {}"
    # parse queries and targets

    def process_index_example(index_entry):
        iimage_path, iid = index_entry.split()
        iimage_path = "/".join(iimage_path.split("/")[1:])  # remove the first 'imgnet/' as we are using dtin now
        iimage_path = os.path.join("./data/dtin/", iimage_path)

        ima = process_img(iimage_path, 224)
        return IndexExample(iid=int(iid), iimage=ima, itokens=null_tokens)

    def process_query(query_entry, domain_id, domain):
        qimg_path, class_id = query_entry.split()
        qimg_path = "/".join(qimg_path.split("/")[1:])  # remove the first 'imgnet/' as we are using dtin now
        qimage_path = os.path.join("./data/dtin/", qimg_path)
        target_iid = domain_id * 1000 + int(class_id)
        qid = qimg_path.split("/")[-1]

        qtext = query_text_template.format(domain)
        qimage = process_img(qimage_path, 224)
        qtokens = np.array(tokenizer(qtext))
        return QueryExample(qid=qid + "-class-" + class_id + "-to-" + domain, qtokens=qtokens, qimage=qimage, target_iid=target_iid, retrieved_iids=[], retrieved_scores=[])

    
    domain_id = all_domains.index(target_domain)
    with ThreadPoolExecutor() as executor:
        print("Preparing query examples...")
        query_futures = {executor.submit(process_query, query_entry, domain_id, target_domain): (query_entry, domain_id, target_domain) for query_entry in query_entries}

        with tqdm(total=len(query_entries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    print("Prepared query examples.")

    print("Preparing target examples...")

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_entry): index_entry for index_entry in index_entries}

        with tqdm(total=len(index_entries), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

    # import pdb; pdb.set_trace()
    print("Prepared index examples.")
    return eval_dataset
