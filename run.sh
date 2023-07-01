python -u main.py --data_path=micro_video --data_type=time --dataset=$1 --model=$2 --epochs=600 --decay=$3 --lr=$4 --alpha=$5 --step_size=$6 --group_num=$7 --period=$8 --epsilon=$9 --env_k=$10 --gamma=$11 --predict=$12 --gpu=$13 --log=$14 --load=$15 > ./log/LightGCN_$1_$2_$3decay_$4lr_$5alpha_$6stepsize_$7group_$8stage_$9epsilon_$10envk_$12gamma_$12predict_$14.txt 2>&1 &


# sh run.sh micro_video lgn 1e-3 0.001 0.3 0.17 3 8 0.3 1 4 0.2 0 log_0 0
