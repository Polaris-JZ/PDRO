python -u inference.py --data_path=micro_video/ --data_type=time --dataset=$1 --model=$2 --epochs=600 --decay=$3 --lr=$4 --alpha=$5 --step_size=$6 --group_num=$7 --period=$8 --epsilon=$9 --env_k=$10 --gamma=$11 --predict=$12 --gpu=$13 --log=$14 --load=$15 > ./log/LightGCN_infer_loss_$1_$2_$3decay_$4lr_$5alpha_$6stepsize_$7group_$8stage_$9epsilon_$10envk_$11gamma_$12predict_$14load_$13.txt 2>&1 &

# wd(1e-4,1e-3) lr(0.001,0.01) layer(3,4,5) dim(32,64,128) drop(0,0.1,0.5)



# sh inference.sh micro_video lgn 1e-3 0.001 0.3 0.17 5 8 0.3 1 4 0.2 0 log_0 1
