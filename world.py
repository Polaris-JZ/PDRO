import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "/storage/jjzhao/jujia_ws/PDRO"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['micro_video', 'kuai', 'amazon-book']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.batch_size
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['num_ng'] = args.num_ng
config['group_num'] = args.group_num
config['A_split'] = False
config['bigdata'] = False

config['alpha'] = args.alpha
config['step_size'] = args.step_size
config['log'] = args.log
config['period'] = args.period
config['p'] = args.p
config['epsilon'] = args.epsilon
config['env_k'] = args.env_k
config['predict'] = args.predict
config['gamma'] = args.gamma
config['act_func'] = args.act_func

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
# if dataset not in all_dataset:
#     raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)