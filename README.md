<br>
<p align="center">
<h1 align="center"><strong>StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling</strong></h1>
  <p align="center">
    <a href='https://github.com/kellyiss/' target='_blank'>Meng Wei*</a>&emsp;
    <a href='https://bryce-wan.github.io/' target='_blank'>Chenyang Wan*</a>&emsp;
    <a href='https://scholar.google.com/citations?user=CKWKIscAAAAJ&hl=en' target='_blank'>Xiqian Yu*</a>&emsp;
    <a href='https://tai-wang.github.io/' target='_blank'>Tai Wang*‡</a>&emsp;
    <a href='https://yuqiang-yang.github.io/' target='_blank'>Yuqiang Yang</a>&emsp;
    <a href='https://scholar.google.com/citations?user=-zT1NKwAAAAJ&hl=en' target='_blank'>Xiaohan Mao</a>&emsp;
    <a href='https://zcmax.github.io/' target='_blank'>Chenming Zhu</a>&emsp;
    <a href='https://wzcai99.github.io/' target='_blank'>Wenzhe Cai</a>&emsp;
    <a href='https://hanqingwangai.github.io/' target='_blank'>Hanqing Wang</a>&emsp;
    <a href='https://yilunchen.com/about/' target='_blank'>Yilun Chen</a>&emsp;
    <a href='https://xh-liu.github.io/' target='_blank'>Xihui Liu†</a>&emsp;
    <a href='https://oceanpang.github.io/' target='_blank'>Jiangmiao Pang†</a>&emsp;
    <br>
    Shanghai AI Laboratory&emsp;The University of Hong Kong&emsp;Zhejiang University&emsp;Shanghai Jiao Tong University&emsp;
  </p>
</p>

<div id="top" align="center">

[![Project](https://img.shields.io/badge/Project-%23FF9E7D?style=flat&logoColor=%23A2D2FF)](https://streamvln.github.io/)
[![arXiv](https://img.shields.io/badge/Arxiv-%23A2D2FF?style=flat&logoColor=%23FF9E7D)](http://arxiv.org/abs/2507.05240)

</div>

## 🏠 About
<strong><em>StreamVLN</em></strong> generates action outputs from continuous video input in an online, multi-turn dialogue manner. Built on LLaVA-Video as the foundational Video-LLM, we extend it for interleaved vision, language, and action modeling. For both effective context modeling of long sequence and efficient computation for real-time interaction, StreamVLN has: (1) a fast-streaming dialogue context with a sliding-window KV cache; and (2) a slow-updating memory via token pruning.
<div style="text-align: center;">
    <img src="assets/teaser.gif" width=100% >
</div>

## 🛠 Getting Started
We test under the following environment:
* Python 3.9
* Pytorch 2.1.2
* CUDA Version 12.1 

1. **Preparing  a conda env with `Python3.9` & Install habitat-sim and habitat-lab**
    ```bash
    conda create -n streamvln python=3.9
    conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
    git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e habitat-lab  # install habitat_lab
    pip install -e habitat-baselines # install habitat_baselines
    ```

2. **Clone this repository**
    ```bash
    git clone https://github.com/OpenRobotLab/StreamVLN.git
    cd StreamVLN
    ```

3. **Data Preparation**

    Please prepare the data following [HowTo use common supported datasets with Habitat-Sim.](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)

    The data folder should follow this structure:
    ```shell
    data/
    ├── datasets/
    │   └── vln/
    │       └── mp3d/               
    │           ├── r2r/            
    │           └── rxr/            
    └── scene_datasets/
        └── mp3d/                   
            ├── 17DRP5sb8fy/        
            ├── 1LXtFkjw3qL/
            └── ...                 
    ```

## 🏆 Model Zoo

We provide two model checkpoints for different use cases:

- **Benchmark Reproduction**  
  Use this [checkpoint](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln) to reproduce results on the VLN-CE benchmark.

- **Real-World Deployment**  
  This [checkpoint](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world) is recommended for deployment on physical robots.

  We made two modifications:  
  1. **Remove redundant initial turn actions**: The initial left/right turns not mentioned in the instructions are removed for better instruction alignment.  
  2. **Trajectory safety**: Enhanced obstacle avoidance ensures more reliable navigation in real-world environments.


## 🤖 Evaluation

To perform multi-GPU evaluation with key-value cache support, simply run:

```bash
sh streamvln_eval_multi_gpu.sh
```


## 📝 TODO List

- ✅ Release the arXiv paper (July 2025)
- ✅ Provide inference scripts and model checkpoints
- ⏳ Release training code and configurations (Coming Soon)
- ⏳ Release Training Data


## 🔗 Citation

If you find our work helpful, please consider starring this repo 🌟 and cite:

```bibtex
@misc{wei2025streamvlnstreamingvisionandlanguagenavigation,
      title={StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling}, 
      author={Meng Wei and Chenyang Wan and Xiqian Yu and Tai Wang and Yuqiang Yang and Xiaohan Mao and Chenming Zhu and Wenzhe Cai and Hanqing Wang and Yilun Chen and Xihui Liu and Jiangmiao Pang},
      year={2025},
      eprint={2507.05240},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.05240}, 
}
```

## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements

This repo is based on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).