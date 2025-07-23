<h1 align="center">MomentSeeker: A Comprehensive Benchmark and A Strong Baseline For Moment Retrieval Within Long Videos</h1>
<p align="center">
    
<a href="https://yhy-2000.github.io/MomentSeeker/">
    <img alt="Homepage" src="https://img.shields.io/badge/🌐 Homepage-MomentSeeker-blue">
</a>
    <a href="https://arxiv.org/abs/2502.12558">
            <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3A2406.04264-B31B1B.svg">
    </a>
    <a href="https://huggingface.co/datasets/avery00/MomentSeeker">
        <img alt="Build" src="https://img.shields.io/badge/🤗 Dataset-MomentSeeker Benchmark-yellow">
    </a>
</p>

This repo contains the annotation data for the paper "[MomentSeeker: A Comprehensive Benchmark and A Strong Baseline For Moment Retrieval Within Long Videos](https://arxiv.org/abs/2502.12558)".



## 🔔 News:
- 🥳 2025/03/25: We have released the evaluation code of MomentSeeker. 🔥
- 🥳 2025/03/07: We have released the MomentSeeker [Benchmark](https://huggingface.co/datasets/avery00/MomentSeeker) and [Paper](https://arxiv.org/abs/2502.12558)! 🔥

## License
Our dataset is under the CC-BY-NC-SA-4.0 license.

⚠️ If you need to access and use our dataset, you must understand and agree: **This dataset is for research purposes only and cannot be used for any commercial or other purposes. The user assumes all effects arising from any other use and dissemination.**

We do not own the copyright of any raw video files. Currently, we provide video access to researchers under the condition of acknowledging the above license. For the video data used, we respect and acknowledge any copyrights of the video authors. Therefore, for the movies, TV series, documentaries, and cartoons used in the dataset, we have reduced the resolution, clipped the length, adjusted dimensions, etc. of the original videos to minimize the impact on the rights of the original works. 

If the original authors of the related works still believe that the videos should be removed, please contact hyyuan@ruc.edu.cn or directly raise an issue.


## Introduction

We present MomentSeeker, a comprehensive benchmark to evaluate retrieval models' performance in handling general long-video moment retrieval (LVMR) tasks. MomentSeeker offers three key advantages. First, it incorporates long videos of over 500 seconds on average, making it the first benchmark specialized for long-video moment retrieval. Second, it covers a wide range of task categories (including Moment Search, Caption Alignment, Image-conditioned Moment Search, and Video-conditioned Moment Search) and diverse application scenarios (e.g., sports, movies, cartoons, and ego), making it a comprehensive tool for assessing retrieval models' general LVMR performance. Additionally, the evaluation tasks are carefully curated through human annotation, ensuring the reliability of assessment. We further fine-tune an MLLM-based LVMR retriever on synthetic data, which demonstrates strong performance on our benchmark. The checkpoint will release soon.



![Illustrative overview of our MomentSeeker dataset.](https://cdn-uploads.huggingface.co/production/uploads/66d916a7b86f0d569aa19b60/ff-9bFKlN466wElhiA4Wi.png)


## License
Our dataset is under the CC-BY-NC-SA-4.0 license.

⚠️ If you need to access and use our dataset, you must understand and agree: **This dataset is for research purposes only and cannot be used for any commercial or other purposes. The user assumes all effects arising from any other use and dissemination.**

We do not own the copyright of any raw video files. Currently, we provide video access to researchers under the condition of acknowledging the above license. For the video data used, we respect and acknowledge any copyrights of the video authors. Therefore, for the movies, TV series, documentaries, and cartoons used in the dataset, we have reduced the resolution, clipped the length, adjusted dimensions, etc. of the original videos to minimize the impact on the rights of the original works. 

If the original authors of the related works still believe that the videos should be removed, please contact hyyuan@ruc.edu.cn or directly raise an issue.


## Evaluation
> Before you access our dataset, we kindly ask you to thoroughly read and understand the license outlined above. If you cannot agree to these terms, we request that you refrain from downloading our video data.

The JSON file provides candidate videos for each question. The candidates can be ranked, and metrics such as Recall@1 and MAP@5 can be computed accordingly.


We evaluate the video models (LanguageBind and InternVideo2) using an input of uniformly sampled 8 frames, while COVR follows its default setting of 15 frames. For image models, we use the temporally middle frame as the video input. Additionally, we provide the evaluation code as a reference. To reproduce our results, users should follow the respective original repositories to set up a conda environment, download the model weights, and then run our code.


## Hosting and Maintenance
The annotation files will be permanently retained. 

If some videos are requested to be removed, we will replace them with a set of video frames sparsely sampled from the video and adjusted in resolution. Since **all the questions in MomentSeeker are only related to visual content** and do not involve audio, this will not significantly affect the validity of MomentSeeker (most existing MLLMs also understand videos by frame extraction).

If even retaining the frame set is not allowed, we will still keep the relevant annotation files, and replace them with the meta-information of the video, or actively seek more reliable and risk-free video sources.





## Citation

If you find this repository useful, please consider giving a star 🌟 and citation

```
@misc{yuan2025momentseekercomprehensivebenchmarkstrong,
      title={MomentSeeker: A Comprehensive Benchmark and A Strong Baseline For Moment Retrieval Within Long Videos}, 
      author={Huaying Yuan and Jian Ni and Yueze Wang and Junjie Zhou and Zhengyang Liang and Zheng Liu and Zhao Cao and Zhicheng Dou and Ji-Rong Wen},
      year={2025},
      eprint={2502.12558},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.12558}, 
}
```
