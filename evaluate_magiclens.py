# source activate /share/huaying/envs/magiclens/
import sys
sys.path.append('./magiclens/')
from data_utils import *
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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


def process_img(image_path: str, size: int) -> np.ndarray:
    """Process a single image to 224x224 and normalize."""
    img = Image.open(image_path).convert("RGB")
    ima = jnp.array(img)[jnp.newaxis, ...] # [1, 224, 224, 3]
    ima = ima / (ima.max() + 1e-12)  # avoid 0 division
    ima = jax.image.resize(ima, (1, size, size, 3), method='bilinear')
    return np.array(ima)


def process_video(video_path: str, size: int) -> np.ndarray:

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

    ima = jnp.array(frame)[jnp.newaxis, ...] # [1, 224, 224, 3]
    ima = ima / (ima.max() + 1e-12)  # avoid 0 division
    ima = jax.image.resize(ima, (1, size, size, 3), method='bilinear')
    return np.array(ima)



def build_dataset(file: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(file.split('/')[-1].split('.')[0])
    queries = json.load(open(file))

    index_video_clips = []
    for dic in queries:
        index_video_clips.extend([clip['frame_path'] for clip in dic['candidate_video_list']])

    index_video_clips = list(set(index_video_clips))


    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_video_clip_path):
        ima = process_img(index_video_clip_path, 224)
        return IndexExample(iid=index_video_clip_path, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qtext = ' '.join(query["qry_text"].split()[:30])
        # qtext = query["qry_text"]
        qtokens = tokenizer(qtext)

        if query["qry_img_path"]!='': # ti2v
            ima = process_img(query["qry_img_path"], 224)
        elif query["qry_video_path"]!='': # tv2v
            ima = process_video(query["qry_video_path"], 224)
        else: # t2v
            ima = None

        gt_clips = [query['candidate_video_list'][i]['frame_path'] for i in query['gt_indices']]
        return QueryExample(qid=query["qry_text"], qtokens=qtokens, qimage=ima, target_iid=gt_clips,candidate_video_list=[], retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_video_clip): index_video_clip for index_video_clip in index_video_clips}

        with tqdm(total=len(index_video_clips), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset



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





def compute_embeddings(samples, model):
    video_paths = set(sample['tgt_video_path'] for sample in samples)
    video_embeddings = {path: model.compute_video_embedding(path) for path in video_paths}

    qry_text_embeddings = [model.compute_text_embedding(sample['qry_text']) for sample in samples]
    qry_image_embeddings = [model.compute_image_embedding(sample['qry_image_path']) for sample in samples]
    qry_video_embeddings = [model.compute_video_embedding(sample['qry_video_path']) for sample in samples]

    return video_embeddings, qry_text_embeddings, qry_image_embeddings, qry_video_embeddings



def compute_similarity(qry_embeddings, tgt_embeddings):
    similarities = np.dot(qry_embeddings, tgt_embeddings.T)
    return similarities

def compute_recall(similarities, gt_indices, k=1):
    recall = 0
    for i, sim in enumerate(similarities):
        top_k_indices = np.argsort(sim)[-k:]
        if gt_indices[i] in top_k_indices:
            recall += 1
    return recall / len(gt_indices)

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str, default="/share/huaying/long_video/video_retriever/benchmark/retrieval/webvid/eval_tv2v.jsonl")
    parser.add_argument("--model_path", type=str, default="./magiclens/models/magic_lens_clip_base.pkl")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    tokenizer = clip_tokenizer.build_tokenizer()
    model, params = load_model(args.model_size, args.model_path)

    output_dir = './languagebind/benchmark/test_grounding/result/magiclens/'
    os.makedirs(output_dir,exist_ok=True)
    
    benchmark_dir = './languagebind/benchmark/test_grounding/'
    for file in os.listdir(benchmark_dir):
        # try:
            
            output_file = os.path.join(output_dir, f"{file.split('.')[0]}_results.json")
            # if os.path.exists(output_file):
            #     continue

            if 'sports' not in file or 't2v' not in file:
                continue
            
            print('processing',file)


            if not file.endswith('.json') or 'msrvtt' in file:
                continue

            eval_dataset = build_dataset(os.path.join(benchmark_dir,file), tokenizer)

            # inference index:
            index_embeddings = []
            print("Inference index...")
            for i in tqdm(range(0, len(eval_dataset.index_examples), args.batch_size)):
                batch = eval_dataset.index_examples[i : i + args.batch_size]
                iids = [i.iid for i in batch]
                iimages = jnp.concatenate([i.iimage for i in batch], axis=0)
                itokens = jnp.concatenate([i.itokens for i in batch], axis=0)
                iembeds = model.apply(params, {"ids": itokens, "image": iimages})["multimodal_embed_norm"]
                index_embeddings.append(iembeds)
            index_embeddings = jnp.concatenate(index_embeddings, axis=0)

            print("Inference queries...")
            for i in tqdm(range(0, len(eval_dataset.query_examples), args.batch_size)):
                batch = eval_dataset.query_examples[i : i + args.batch_size]
                if batch[0].qimage is not None:
                    qimages = jnp.concatenate([q.qimage for q in batch], axis=0)
                    qtokens = jnp.concatenate([q.qtokens for q in batch], axis=0)
                    qembeds = model.apply(params, {"ids": qtokens, "image": qimages})["multimodal_embed_norm"]
                else:
                    qtokens = jnp.concatenate([q.qtokens for q in batch], axis=0)
                    qimages = None
                    qembeds = model.apply(params, {"ids": qtokens, "image": qimages})["multimodal_embed_norm"]

                similarity_scores = jnp.dot(qembeds, index_embeddings.T)
                # get top 50 by similarity (by default)
                top_k_indices = jnp.argsort(similarity_scores, axis=1)[:, -50:][:, ::-1]
                top_k_iids = [
                    [eval_dataset.index_examples[idx].iid for idx in top_k]
                    for top_k in top_k_indices
                ]

                # gather scores for the top_k
                top_k_scores = [
                    similarity_scores[i, tk].tolist() for i, tk in enumerate(top_k_indices)
                ]

                # update the query_example with the retrieved results
                for k, q_example in enumerate(batch):
                    q_example.retrieved_iids = top_k_iids[k]
                    q_example.retrieved_scores = top_k_scores[k]
                    eval_dataset.query_examples[i + k] = q_example
            
            
            eval_dataset.evaluate_recall(output_dir)
            eval_dataset.evaluate_map(output_dir)
            eval_dataset.write_to_file(output_dir)

        # except Exception as e:
        #     print(e)

if __name__ == "__main__":
    main()

    

