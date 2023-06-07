import world
import utils
import torch
from torch.autograd import grad
from torch import nn, optim
import numpy as np
import math
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import dataloader
from parse import parse_args
import register
import torch.utils.data as data
from model import PopPredictor
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

# construct the train and test datasets
args = parse_args()
# dataset = dataloader.Loader(path = args.data_path, group_num = args.group_num)
dataset = dataloader.DROData(path = args.data_path, group_num = args.group_num) 
train_loader = data.DataLoader(dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

predictor = PopPredictor()
predictor = predictor.to(world.device)
bpr = utils.BPRLoss(Recmodel, predictor, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

config = world.config

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    print("not enable tensorflowboard")

try:
    best_recall = 0
    best_epoch = 0
    recall_list = []
    cnt = 0

    stage_num = config['period']
    loss_list = [0 for _ in range(config['group_num'])]
    w_list = [torch.ones(1).to(world.device) for _ in range(config['group_num'])]

    loss_ge = torch.zeros((config['group_num'], stage_num)).cuda()
    grad_ge = torch.zeros((config['group_num'], stage_num, Recmodel.embedding_item.weight.reshape(-1).size(0))).cuda() 

    beta_e = torch.softmax(torch.tensor([math.exp((e-stage_num)*config['env_k']) for e in range(stage_num)]), dim=0).unsqueeze(0).unsqueeze(-1).cuda()
    for epoch in range(world.TRAIN_epochs):

        torch.cuda.empty_cache()

        Recmodel.train()
        bpr_: utils.BPRLoss = bpr
        train_loader.dataset.get_pair_bpr()
        aver_loss = 0.
        idx = 0
        for batch_users, batch_pos, batch_neg, batch_stage, batch_pos_inter, batch_neg_inter in train_loader:
            batch_users = batch_users.to(world.device)
            batch_pos = batch_pos.to(world.device)
            batch_neg = batch_neg.to(world.device)
            batch_stage = batch_stage.to(world.device)
            batch_pos_inter = torch.stack(batch_pos_inter)
            batch_neg_inter = torch.stack(batch_neg_inter)
            batch_pos_inter = batch_pos_inter.to(world.device)
            batch_neg_inter = batch_neg_inter.to(world.device)
            
            loss, pos_score = bpr.cal_bpr(batch_users, batch_pos, batch_neg, batch_stage, batch_pos_inter, batch_neg_inter, predictor)

            # get the group
            sorted_loss, sorted_indices = torch.sort(pos_score, descending=True)
            part_len = loss.size(0) // world.config['group_num']
            attribute = torch.zeros_like(sorted_indices)
            for g_idx in range(world.config['group_num']):
                attribute[g_idx*part_len:(g_idx+1)*part_len] = g_idx
            batch_group = torch.zeros_like(attribute)
            batch_group[sorted_indices] = attribute
            batch_group = batch_group.to(world.device)


            for name, param in Recmodel.named_parameters():
                if name == 'embedding_item.weight':
                    for g_idx in range(config['group_num']):
                        for e_idx in range(stage_num):
                            indices = (batch_group==g_idx)&(batch_stage==e_idx)
                            de = torch.sum(indices)
                            loss_single = torch.sum(loss*(indices).cuda()) # +1e-16
                            grad_single = grad(loss_single, param, retain_graph=True)[-1].reshape(-1) # linear layer input*output
                            grad_single = grad_single/(grad_single.norm()+1e-16) * torch.pow(loss_single/(de+1e-16), config['p']) 
                            loss_ge[g_idx,e_idx] = loss_single
                            grad_ge[g_idx,e_idx] = grad_single #/(de+1e-16)
            
            # performance term
            de = torch.tensor([torch.sum(batch_group==g_idx) for g_idx in range(config['group_num'])]).cuda()
            loss_ = torch.sum(loss_ge,dim=1)
            loss_ = loss_/(de+1e-16)
            
            # shifting trend term
            trend_ = torch.zeros(config['group_num']).cuda()
            sum_gie = torch.mean(grad_ge * beta_e, dim=[0,1])
            for g_idx in range(config['group_num']):
                g_j = torch.mean(grad_ge[g_idx],dim=0) # sum up the env gradient for group 
                trend_[g_idx] = g_j@sum_gie

            loss_ = loss_ * (1-config['epsilon']) + trend_ * config['epsilon']

            for i in range(config['group_num']):
                if len(torch.nonzero(batch_group==i)) == 0:
                    loss_group = loss_list[i]
                else:
                    loss_group = loss_[i]

                loss_list[i] = (1 - config['alpha']) * loss_list[i] + config['alpha'] * loss_group
                update_factor = config['step_size'] * loss_list[i]
                w_list[i] = w_list[i] * torch.exp(update_factor)

            sum_ = sum(w_list)
            w_list = [i / sum_ for i in w_list]            
            loss = torch.zeros(1).to(world.device)
            for i in range(config['group_num']):
                final_loss = w_list[i] * loss_list[i]
                loss += final_loss
            loss = torch.sum(loss)
            aver_loss += loss
            idx += 1
            loss.backward()
            bpr_.opt.step()
            bpr_.opt.zero_grad()
            w_list = [i.detach() for i in w_list]
            loss_list = [i.detach() for i in loss_list]
            loss_ge = loss_ge.detach()
            grad_ge = grad_ge.detach()

        print(f'epoch {epoch} group loss:{loss_list}')
        print(f'epoch {epoch} group weight:{w_list}')
        aver_loss = aver_loss / idx
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {aver_loss.item()}')


        if epoch >= 20 and (epoch+1) % 5 == 0:
            v_results = Procedure.Test(dataset, Recmodel, predictor, epoch, w, world.config['multicore'], 0)
            Procedure.print_results(None, v_results, None)
            t_results = Procedure.Test(dataset, Recmodel, predictor, epoch, w, world.config['multicore'], 1)
            Procedure.print_results(None, None, t_results)
            if v_results[1][0] > best_recall:
                best_epoch = epoch
                best_recall = v_results[1][0]
                best_v, best_t = v_results, t_results
                torch.save(Recmodel.state_dict(), weight_file)
            if epoch == 50:
                recall_list.append((epoch, v_results[1][0]))
            # early stopping
            if epoch > 50:
                recall_list.append((epoch, v_results[1][0]))
                if v_results[1][0] < best_recall:
                    cnt += 1
                else:
                    cnt = 1
                if cnt >= 8:
                    break
        
    print("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
    print("Validation:")
    Procedure.print_results(None, best_v, None)
    print("Test:")
    Procedure.print_results(None, None, best_t)

finally:
    if world.tensorboard:
        w.close()