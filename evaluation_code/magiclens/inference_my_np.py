import os
import json
import numpy as np
import pickle
from typing import Dict, List
from argparse import ArgumentParser
from model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from flax import serialization
from tqdm import tqdm
from PIL import Image

def load_model(model_size: str, model_path: str):
    model = MagicLens(model_size)
    dummy_input = {
        "ids": np.ones((1, 1, 77), dtype=np.int32),
        "image": np.ones((1, 224, 224, 3), dtype=np.float32),
    }
    params = model.init(dummy_input)
    with open(model_path, "rb") as f:
        model_bytes = pickle.load(f)
    params = serialization.from_bytes(params, model_bytes)
    return model, params

def process_video(video_path: str) -> np.ndarray:
    """
    可根据实际需求对视频进行抽帧和预处理，这里仅返回一个224x224的占位帧
    """
    # TODO: 添加真实视频预处理逻辑
    dummy_frame = np.ones((1, 224, 224, 3), dtype=np.float32)
    return dummy_frame

def run_inference(input_data: Dict, model, params, tokenizer, top_k: int = 50) -> List[str]:
    """
    input_data:
    {
        "qry_text": "...",
        "qry_img_path": "",
        "qry_video_path": "/path/to/qry.mp4",
        "tgt_video_path": ["/path/to/gt.mp4", "/path/to/other.mp4", ...]
    }
    """
    q_tokens = np.array(tokenizer(input_data["qry_text"]))
    q_video = process_video(input_data["qry_video_path"])
    q_embed = model.apply(params, {"ids": q_tokens, "image": q_video})["multimodal_embed_norm"]

    tgt_video_paths = input_data["tgt_video_path"]
    scores = []
    for v_path in tgt_video_paths:
        v_tensors = process_video(v_path)
        v_embed = model.apply(params, {"ids": q_tokens, "image": v_tensors})["multimodal_embed_norm"]
        score = np.dot(q_embed, v_embed.T).item()
        scores.append((v_path, score))

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [item[0] for item in ranked]

def calculate_recall_at_1(ranked_list: List[str], ground_truth: str) -> float:
    """
    计算 recall@1 分数
    """
    return 1.0 if ranked_list[0] == ground_truth else 0.0

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str, default="/share/huaying/long_video/video_retriever/benchmark/retrieval/corpus_level_tv2v_youtube/eval_tv2v.jsonl")
    parser.add_argument("--model_path", type=str, default="/share/huaying/long_video/video_retriever/baselines/magiclens/models/magic_lens_clip_base.pkl")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        dic_li = json.load(f)
        recall_at_1_list = []
        tokenizer = clip_tokenizer.build_tokenizer()
        model, params = load_model(args.model_size, args.model_path)
        for input_data in tqdm(dic_li):
            ranked_list = run_inference(input_data, model, params, tokenizer, top_k=args.top_k)

            ground_truth = input_data["tgt_video_path"][0]
            recall_at_1 = calculate_recall_at_1(ranked_list, ground_truth)
            recall_at_1_list.append(recall_at_1)

        recall_at_1_avg = sum(recall_at_1_list) / len(recall_at_1_list)
        print(f"Recall@1: {recall_at_1_avg}")

if __name__ == "__main__":
    main()


    