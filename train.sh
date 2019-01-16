now=$(date +"%Y%m%d_%H%M%S")
CUDA_VISIBLE_DEVICES=7 python  ppo_train.py --dm-size 5 --edge-num 10 \
    --struct fc --learning-rate 0.0001 --gen-rule constant --num-threads 10\
    --min-batch-size 20 --cyc-len 10 \
    --model-path "learned_models/ppo/constant-fc/499.p" \
      2>&1 | tee log/ppo-$now.txt
