# Popularity-aware Distributionally Robust Optimization for Recommendation System
This is the pytorch implementation of our paper
> Popularity-aware Distributionally Robust Optimization for Recommendation System

## Environment
- Anaconda 3
- Python 3.8.12
- Pytorch 1.7.0
- Numpy 1.21.2

## The Micro-video Dataset
The Micro-video dataset, sourced from the Huawei micro-video App integrated into Huawei mobile phones, comprises a collection of user-item interactions spanning a month. This dataset encompasses a wide range of interactions with diverse micro-videos.

| #user | #item | #Interaction |
|:--------:|:--------:|:--------:|
| 25,871 | 44,503 | 210,550 |
## Training
Run the PDRO on Micro-video dataset:
```bash
sh run.sh micro_video lgn 1e-3 0.001 0.3 0.17 3 8 0.3 1 4 0.2 0 log_0 0
```

## Inference (Including Group Evluation)
Infer the PDRO on Micro-video dataset:
```bash
sh inference.sh micro_video lgn 1e-3 0.001 0.3 0.17 5 8 0.3 1 4 0.2 0 log_0 1
```

This implementation is based on LightGCN.