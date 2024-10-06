# Zero-Shot Video Question Answering via Frozen Bidirectional Language Models

[Webpage](https://antoyang.github.io/frozenbilm.html) • [Paper](https://arxiv.org/abs/2206.08155) 

![Teaser](https://antoyang.github.io/img/frozenbilm-header.png)

FrozenBiLM is a new model for video question answering that builds on a frozen bidirectional language model. FrozenBiLM excels in settings without any manual annotation (zero-shot) or with limited training data (few-shot), while performing competitively when trained on standard datasets (fully-supervised).

This repository provides the code for our Listen Then See paper (CVPR 2024), including:
- Environment setup
- Data downloading instructions
- Data preprocessing and visual feature extraction scripts
- Pretrained checkpoints
- Training and evaluation scripts for cross-modal training, downstream fully-supervised, few-shot and zero-shot VideoQA, including various baselines
- VideoQA demo script

## Setup
To install requirements, run:
```
conda create -f env.yml
conda activate frozenbilm
```
You may fill the global paths in `args.py`.   
To use a given text-pretrained language model, you should download the corresponding weights from the Hugging Face Hub and put them in `TRANSFORMERS_CACHE`.

## Quick Start
If you wish to start VideoQA training or inference quickly.

### Download preprocessed data, visual features and checkpoints
To download pretrained checkpoints, pre-processed data, ASR and visual features, run:
```
bash download/download_checkpoints.sh <MODEL_DIR>
bash download/download_downstream.sh <DATA_DIR>
```
If you have issues with gshell, you can access the processed data [here](https://drive.google.com/drive/folders/1ED2VcFSxRW9aFIP2WdGDgLddNTyEVrE5?usp=sharing)  and the checkpoints [here](https://drive.google.com/drive/folders/10Vosd_h6afVf-OSZmwVeTCQReZwpAUJT?usp=sharing).  
It requires about 8GB for the models, and 12GB for the data.  
Note that most pretrained checkpoints only contain updated parameters due to storage limitations (and not the frozen parameters).  
This means you have to make sure that you have properly downloaded weights from Hugging Face for the language model of your choice when using a provided checkpoint.  
For completeness, `frozenbilm.pth`, `frozenbilm_bertbase_noadapter.pth` and `frozenbilm_bertlarge_noadapter.pth` contain all parameters.  
Also note that due to storage issue, we do not host publicly visual features for the WebVid10M dataset.   


## Available checkpoints

| Training data | LSMDC | iVQA | MSRVTT-QA | MSVD-QA | ActivityNet-QA | TGIF-QA | How2QA | TVQA | url | size |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| WebVid10M | 51.5 | 26.8 | 16.7 | 33.8 | 25.9 | 41.9 | 58.4 | 59.7 | [Drive](https://drive.google.com/file/d/1-_mUTxSjQj-NZ-le0-mDUftaikB_2nsU/view?usp=sharing)    | 3.7GB (inc. frozen weights)      |
| WebVid10M + LSMDC | 63.5 | | | | | | | | [Drive](https://drive.google.com/file/d/1y5YCOvfonDf1OBTiAdys_Dn9yOoViD1t/view?usp=sharing)    | 114MB      |
| WebVid10M + iVQA | | 39.6 | | | | | | | [Drive](https://drive.google.com/file/d/190isvIe8DmrzTuBad-hNMiQkgLA0JewM/view?usp=sharing)    | 114MB      |
| WebVid10M + MSRVTT-QA | | | 47.0 | | | | | | [Drive](https://drive.google.com/file/d/1RKFK2hoIbSaDRus8Ml57_xGG9BuRrhWp/view?usp=sharing)    | 114MB      |
| WebVid10M + MSVD-QA | | | | 54.8 | | | | | [Drive](https://drive.google.com/file/d/1Jb9egGCZaH30ZBaRz1hhK2x2cDkCsNEd/view?usp=sharing)    | 114MB      |
| WebVid10M + ActivityNet-QA | | | | | 43.2 | | | | [Drive](https://drive.google.com/file/d/1etIAIo086MIGo2cYVTy4hOFt3DkKJpYv/view?usp=sharing)    | 114MB      |
| WebVid10M + TGIF-QA | | | | | | 68.6 | | | [Drive](https://drive.google.com/file/d/1PBzLGW3uWdm92kmy9OfLILIwJSl9ifAI/view?usp=sharing)    | 114MB      |
| WebVid10M + How2QA| | | | | | | 86.3 | | [Drive](https://drive.google.com/file/d/1mJnO2CUUuyfQ6ic2bU6PbuivmArvXAOT/view?usp=sharing)    | 114MB      |
| WebVid10M + TVQA | | | | | | | | 82.0 | [Drive](https://drive.google.com/file/d/15vsfNJf9UsWbmimibfPhLoHRrbZhF7W6/view?usp=sharing)    | 114MB      |

Note that checkpoints finetuned on 10% or 1% of downstream datasets (few-shot setting) are also made accessible [here](https://drive.google.com/drive/u/8/folders/10Vosd_h6afVf-OSZmwVeTCQReZwpAUJT).  
Variants using a BERT-Base or BERT-Large language model (without adapters) instead of DeBERTa are also present in this folder.

## Cross-modal training

## Zero-shot VideoQA

### Multiple-choice VideoQA

# Export cache
export TRANSFORMERS_CACHE=transformers_cache/

# Predict siq2
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=zssiq --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth

# Finetune siq2
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ftsiq2_test --lr=5e-5 --schedule=linear_with_warmup --load=models/frozenbilm.pth --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size=2 --batch_size_val=8 --max_tokens=256 --epochs=20 


## Licenses
This code is released under the Apache License 2.0.  
The licenses for datasets used in the paper are available at the following links: [iVQA](https://github.com/antoyang/just-ask/blob/main/LICENSE), [MSRVTT-QA](https://github.com/xudejing/video-question-answering/blob/master/LICENSE), [MSVD-QA](https://github.com/xudejing/video-question-answering/blob/master/LICENSE), [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa/blob/master/LICENSE), [How2QA](https://github.com/ych133/How2R-and-How2QA/blob/master/LICENSE) and [TVQA](https://github.com/jayleicn/TVQA/blob/master/LICENSE).

## Citation 
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@inproceedings{agrawal2024listen,
  title={Listen Then See: Video Alignment with Speaker Attention},
  author={Agrawal, Aviral and Lezcano, Carlos Mateo Samudio and Heredia-Marin, Iqui Balam and Sethi, Prabhdeep Singh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2018--2027},
  year={2024}
}
```
