import world
import numpy as np
import torch
import utils
import math
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import pdb
from torch import nn, optim

CORES = multiprocessing.cpu_count() // 2

def get_batch_group(feature, label, dataset):
    group_dict = {}
    group = dataset.group
    for (fea, lab) in zip(feature,label):
        if group[fea[0].item()] not in group_dict:
            group_dict[group[fea[0].item()]] = []
        group_dict[group[fea[0].item()]].append(torch.stack((fea[0], fea[1], lab)))
    # print(group_dict)
    for i in group_dict.keys():
        group_dict[i] = torch.stack(group_dict[i])
    return group_dict

def get_batch_group_bpr(batch_users, batch_pos, batch_neg, dataset):
    group_dict = {}
    group = dataset.group
    for (user, pos, neg) in zip(batch_users, batch_pos, batch_neg):
        if group[user.item()] not in group_dict:
            group_dict[group[user.item()]] = []
        group_dict[group[user.item()]].append(torch.stack((user, pos, neg)))
    # print(group_dict)
    for i in group_dict.keys():
        group_dict[i] = torch.stack(group_dict[i])
    return group_dict
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    #user_list = [1309,1662,1961] # stable
    #user_list = [1941,1909,1799,1662,1626,1624,1600] # unstable
    for index in range(len(topN)):
        # print(f'top {topN[index]}\n')
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                cnt += 1
            # else: 
            #     print('OPS')
#             if i in user_list:
#                 print(f'user {i}')
# #                 print(predictedIndices[i])
# #                 print(GroundTruth[i])
#                 print(userHit / len(GroundTruth[i]))
#                 print(ndcg)
        precision.append(round(sumForPrecision / cnt, 4))
        recall.append(round(sumForRecall / cnt, 4))
        NDCG.append(round(sumForNdcg / cnt, 4))
        MRR.append(round(sumForMRR / cnt, 4))
        
    return precision, recall, NDCG, MRR

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

def Test_group(cnt, dataset, Recmodel, predictor, epoch, w=None, multicore=0, flag=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    if flag == 0:
        testDict = dataset.valid_dict
    else:
        testDict = dataset.test_dict
    item_num = dataset.m_item
    if flag == 0 :
        stage = torch.full((item_num,1), world.config['period'])
    else:
        stage = torch.full((item_num,1), world.config['period']+1)
    item_inter = dataset.itemInter
    item_pop = []
    for i in range(item_num):
        item_pop.append(item_inter[i])
    stage = torch.squeeze(stage)

    item_pop = torch.Tensor(item_pop)
    stage = stage.to(world.device)
    item_pop = item_pop.to(world.device)

    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    # group = dataset.group_user
    group = np.load('/storage/jjzhao/jujia_ws/cikm_huawei/LightGCN_tdro/loss_group.npy', allow_pickle=True).item()
    group_test = {}
    for user in testDict:
        if group[user] == cnt:
            group_test[user] = testDict[user]
    with torch.no_grad():
        users = list(group_test.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [group_test[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            rating = Recmodel.getUsersRating(batch_users_gpu, predictor, world, stage, item_pop)
            #ipdb.set_trace()
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            valid_items = dataset.getUserValidItems(batch_users) # exclude validation items
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if flag:
                for range_i, items in enumerate(valid_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)

            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            # rating_list.append(rating_K.cpu()) # shape: n_batch, user_bs, max_k
            # groundTrue_list.append(groundTrue)
            rating_list.extend(rating_K.cpu()) # shape: n_batch, user_bs, max_k
            groundTrue_list.extend(groundTrue)
        #ipdb.set_trace()
        assert total_batch == len(users_list)
        precision, recall, NDCG, MRR = computeTopNAccuracy(groundTrue_list,rating_list,[10,20,50,100])
        #print_results(None,None,results)
    
        if multicore == 1:
            pool.close()
        return precision, recall, NDCG, MRR

def Test(dataset, Recmodel, predictor, epoch, w=None, multicore=0, flag=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    if flag == 0:
        testDict = dataset.valid_dict
    else:
        testDict = dataset.test_dict
    item_num = dataset.m_item
    if flag == 0 :
        stage = torch.full((item_num,1), world.config['period'])
    else:
        stage = torch.full((item_num,1), world.config['period']+1)
    item_inter = dataset.itemInter
    item_pop = []
    for i in range(item_num):
        item_pop.append(item_inter[i])
    stage = torch.squeeze(stage)

    item_pop = torch.Tensor(item_pop)
    stage = stage.to(world.device)
    item_pop = item_pop.to(world.device)

    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    # results = {'precision': np.zeros(len(world.topks)),
    #            'recall': np.zeros(len(world.topks)),
    #            'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        # try:
        #     assert u_batch_size <= len(users) / 10
        # except AssertionError:
        #     print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            rating = Recmodel.getUsersRating(batch_users_gpu, predictor, world, stage, item_pop)
            #ipdb.set_trace()
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            valid_items = dataset.getUserValidItems(batch_users) # exclude validation items
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if flag:
                for range_i, items in enumerate(valid_items):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)

            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            # rating_list.append(rating_K.cpu()) # shape: n_batch, user_bs, max_k
            # groundTrue_list.append(groundTrue)
            rating_list.extend(rating_K.cpu()) # shape: n_batch, user_bs, max_k
            groundTrue_list.extend(groundTrue)
        #ipdb.set_trace()
        assert total_batch == len(users_list)
        precision, recall, NDCG, MRR = computeTopNAccuracy(groundTrue_list,rating_list,[10,20,50,100])
        #print_results(None,None,results)
    
        if multicore == 1:
            pool.close()
        return precision, recall, NDCG, MRR

def print_epoch_result(results):
    print("Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                    '-'.join([str(x) for x in results['precision']]), 
                                    '-'.join([str(x) for x in results['recall']]), 
                                    '-'.join([str(x) for x in results['ndcg']])))

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
        
def print_results_group(i, loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if i is not None:
        if valid_result is not None: 
            print("[Valid_group{}]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                i,
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]), 
                                '-'.join([str(x) for x in valid_result[2]]), 
                                '-'.join([str(x) for x in valid_result[3]])))
        if test_result is not None: 
            print("[Test_group{}]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                i,
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]), 
                                '-'.join([str(x) for x in test_result[2]]), 
                                '-'.join([str(x) for x in test_result[3]])))

    else:
        if valid_result is not None: 
            print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in valid_result[0]]), 
                                '-'.join([str(x) for x in valid_result[1]]), 
                                '-'.join([str(x) for x in valid_result[2]]), 
                                '-'.join([str(x) for x in valid_result[3]])))
        if test_result is not None: 
            print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                                '-'.join([str(x) for x in test_result[0]]), 
                                '-'.join([str(x) for x in test_result[1]]), 
                                '-'.join([str(x) for x in test_result[2]]), 
                                '-'.join([str(x) for x in test_result[3]])))