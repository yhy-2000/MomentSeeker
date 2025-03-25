# source activate /share/huaying/envs/magiclens/

import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from typing import Dict, List
from argparse import ArgumentParser
from model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from flax import serialization
from tqdm import tqdm

import cv2
import numpy as np


def load_model(model_size: str, model_path: str):
    model = MagicLens(model_size)
    rng = jax.random.PRNGKey(0)
    dummy_input = {
        "ids": jnp.ones((1, 1, 77), dtype=jnp.int32),
        "image": jnp.ones((1, 224, 224, 3), dtype=jnp.float32),
    }
    params = model.init(rng, dummy_input)
    with open(model_path, "rb") as f:
        model_bytes = pickle.load(f)
    params = serialization.from_bytes(params, model_bytes)
    return model, params


def process_video(video_path: str) -> np.ndarray:
    """
    返回视频的中间帧，帧大小为224x224
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"无法读取视频帧: {video_path}")

    cap.release()
    resized_frame = cv2.resize(frame, (224, 224))
    return resized_frame


def run_inference(input_data: Dict, model, params, tokenizer, batch_size: int = 16, top_k: int = 50) -> List[str]:
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
    for v_path in tqdm(tgt_video_paths):
        v_tensors = process_video(v_path)
        v_embed = model.apply(params, {"ids": q_tokens, "image": v_tensors})["multimodal_embed_norm"]
        score = jnp.dot(q_embed, v_embed.T).item()
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
    parser.add_argument("--input_json", type=str, default="/share/huaying/long_video/video_retriever/benchmark/retrieval/webvid/eval_tv2v.jsonl")
    parser.add_argument("--model_path", type=str, default="/share/huaying/long_video/video_retriever/baselines/magiclens/models/magic_lens_clip_base.pkl")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        dic_li = json.load(f)
        recall_at_1_list = []
        for input_data in dic_li:
            tokenizer = clip_tokenizer.build_tokenizer()
            model, params = load_model(args.model_size, args.model_path)
            ranked_list = run_inference(input_data, model, params, tokenizer, top_k=args.top_k)

            ground_truth = input_data["tgt_video_path"][0]
            recall_at_1 = calculate_recall_at_1(ranked_list, ground_truth)
            recall_at_1_list.append(recall_at_1)

        recall_at_1_avg = sum(recall_at_1_list) / len(recall_at_1_list)
        print(f"Recall@1: {recall_at_1_avg}")

if __name__ == "__main__":
    main()

    

