
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/test" ]; then
    mkdir ./logs/test
fi

if [ ! -d "./logs/test/new" ]; then
    mkdir ./logs/test/new
fi

model_name=TimeBridge
seq_len=720
GPU=1
root=./dataset

alpha=0.2
data_name=electricity
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u tune.py \
    --is_training 1 \
    --root_path $root/electricity/ \
    --data_path electricity.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 321 \
    --des 'Exp' \
    --n_heads 32 \
    --d_ff 512 \
    --d_model 512 \
    --ca_layers 2 \
    --pd_layers 1 \
    --ia_layers 1 \
    --attn_dropout 0.1 \
    --num_p 4 \
    --stable_len 4 \
    --alpha $alpha \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --itr 1 | tee logs/test/new/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done