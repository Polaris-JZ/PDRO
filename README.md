# Popularity-aware Distributionally Robust Optimization for Recommendation System
This is the pytorch implementation of our paper
> Popularity-aware Distributionally Robust Optimization for Recommendation System

## Environment
- Anaconda 3
- Python 3.8.12
- Pytorch 1.7.0
- Numpy 1.21.2


## run code
First, change the `ROOT_PATH` in the `world.py` to your own root path

Second, run the PDRO:
```bash
sh run.sh micro_video lgn 1e-3 0.001 0 0.3 0.17 3 8 0.3 1 4 0.2 0 log_0 0
```
