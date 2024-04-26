# Transformer Feature Collapse of Temporal Action Detection via Multi-granularity Semantic Enhancement

## Abstract

Transformer-based models for Temporal Action Detection have achieved significant performance improvements, where the Multi-Head Self-Attention (MHSA) mechanism has played a pivotal role. However, owing to MHSA's tendency to map different patches into similar latent representations, existing methodologies are afflicted with the issue of temporal feature collapse, resulting in high similarity among temporal points and consequently increasing the difficulty in distinguishing action from background. To address the issue, we propose a Multi-granularity Semantic Enhancement (MSE) Block to learn multi-granularity semantic information from different feature spaces. The MSE Block comprises three core components: Local Discriminative Information Modeling (LDM), Global Temporal Information Modeling (GTM), and Adaptive Fusion Module (AFM). LDM facilitates the capture of discriminative information via a multi-scale convolutional group for local detail enhancement, subsequently, GTM employs the MHSA for global temporal context interaction, and AFM adaptively fuses all enhanced features to achieve multi-granularity semantic representation enhancement. Extensive experiments validate the superiority of our method, yielding state-of-the-art performance of 70.5\% on THUMOS14, 39.3\% on HACS, and 36.9\% on ActivityNet-1.3.

## Getting Started

### Environment

- Python 3.8
- PyTorch >= 1.11 (We use version=1.11.0 in our experiments)
- NVIDIA GPU

### Setup

1. Install the required packages by running the following command:

   ```
   pip install -r requirements.txt
   ```

2. Install NMS

   ```
   cd ./libs/utils
   python setup.py install --user
   cd ../..
   ```

## Data Preparation

- We adpot the feature for **THUMOS14** and **ActivityNet-1.3**  datasets from ActionFormer repository ([see here](https://github.com/happyharrycn/actionformer_release)). To use these features, please download them from their link and unpack them into the `./data` folder.
- We adpot the feature for **HACS** dataset from Tridet repository ([see here](https://github.com/dingfengshi/TriDet)). To use these features, please download them from their link and unpack them into the `./data` folder.

## Quick Start

### Training

- Train our model with I3D features on **THUMOS14** dataset. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.

```
python ./train.py ./configs/thumos_i3d.yaml --output reproduce
```

- Train our model with TSP features on **ActivityNet-1.3** dataset. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.

```
python ./train.py ./configs/anet_tsp.yaml --output reproduce
```

- Train our model with SlowFsst features on **HACS** dataset. This will create an experiment folder under *./ckpt* that stores training config, logs, and checkpoints.

```
python ./train.py ./configs/hacs_slowfast.yaml --output reproduce
```

### Inference

We offer pre-trained models for each dataset, which you can download the checkpoint from [Google Driven](https://drive.google.com/drive/folders/1JScEljKDPRxD2v0zYlScVsmayMR3O2XO?usp=drive_link), [Weiyun](https://share.weiyun.com/VZvSGvbY).  The command for test is

```
python eval.py ./configs/CONFIG_FILE PATH_TO_CHECKPOIN
```
