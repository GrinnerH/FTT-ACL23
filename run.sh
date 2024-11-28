# # seasonal-online-bf20 BERT
# python main.py \
#     --gpu 0 \
#     --epoch 50 \
#     --lr 2e-05 \
#     --model_name bert \
#     --root_path  reweight_roll_season/roll_seasonal_data/roll_online_bf20_demo \
#     --bert_path /path/to/chinese-bert-wwm-ext \
#     --data_name seasonal_roll_online_bf20_demo \
#     --data_type roll_online \
#     --split_level seasonal

# # seasonal-online-bf20 BERT+EANN
# python main.py \
#     --gpu 0 \
#     --epoch 50 \
#     --lr 2e-05 \
#     --model_name eann_bert \
#     --root_path reweight_roll_season/roll_seasonal_data/roll_online_bf20_demo \
#     --bert_path /path/to/chinese-bert-wwm-ext \
#     --data_name seasonal_roll_online_bf20_demo \
#     --data_type roll_online \
#     --split_level seasonal

# seasonal-online-bf20 BERT+FTT
python main.py \
    --gpu 0 \
    --epoch 50 \
    --lr 2e-05 \
    --model_name bert \
    --root_path reweight_roll_season/roll_seasonal_user_data/roll_online_bf20_demo_SBERT-sg-0.6_prophet-bsr-3_sw-5.0-0.5-2.0 \
    --bert_path /home/robot/wwh/Huggingface/hub/models--hfl--chinese-bert-wwm-ext \
    --data_name adjusted_data_name \
    --data_type roll_online \
    --split_level seasonal