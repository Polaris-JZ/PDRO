python -u inference.py --data_path=micro_video/ --data_type=time --dataset=$1 --model=$2 --epochs=600 --decay=$3 --lr=$4 --layer=$5 --recdim=$6 --dropout=$7 --alpha=$8 --step_size=$9 --group_num=$10 --period=$11 --epsilon=$12 --env_k=$13 --gamma=$14 --predict=$15 --gpu=$16 --log=$17 --load=$18 > ./log/LightGCN_infer_loss_$1_$2_$3decay_$4lr_$5layer_$6recdim_$7dropout_$8alpha_$9stepsize_$10group_$11stage_$12epsilon_$13envk_$14gamma_$15predict_$18load_$17.txt 2>&1 &

# wd(1e-4,1e-3) lr(0.001,0.01) layer(3,4,5) dim(32,64,128) drop(0,0.1,0.5)



# sh inference.sh micro_video lgn 1e-3 0.001 4 128 0 0.3 0.17 5 8 0.3 1 4 0.2 5 log_10 1
