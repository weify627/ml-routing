now=$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=4 python  supervised_train.py \
    --approach unsupervised \
    --seq-len 10 --cyc-len 20 --dm-size 5 --p 0.5\
    --struct conv --gen-rule "gravity-avg" --lr 0.001 \
      2>&1 | tee log/un-$now.txt
#--model-path "learned_models/ppo_0/24.p" \
