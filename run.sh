python -u main.py --data_path=micro_video --data_type=time --dataset=$1 --model=$2 --epochs=600 --decay=$3 --lr=$4 --dropout=$5 --alpha=$6 --step_size=$7 --group_num=$8 --period=$9 --epsilon=$10 --env_k=$11 --gamma=$12 --predict=$13 --gpu=$14 --log=$15 --load=$16 > ./log/LightGCN_$1_$2_$3decay_$4lr_$5dropout_$6alpha_$7stepsize_$8group_$9stage_$10epsilon_$11envk_$12gamma_$13predict_$15.txt 2>&1 &


# sh run.sh micro_video lgn 1e-3 0.001 0 0.3 0.17 3 8 0.3 1 4 0.2 0 log_0 0
