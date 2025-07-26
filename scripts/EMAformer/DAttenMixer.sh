if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336

random_seed=2021

model_name=DAttenMixer

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 128 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --loss_flag 2\
      --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 128 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --loss_flag 2\
      --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm1.csv \
      --model_id ETTm1'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1 \
      --d_model 128 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 1\
      --d_model 128 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4 \
      --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data weather \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 1 \
      --d_model 128 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8 \
      --des 'Exp' \
      --train_epochs 200\
      --patience 10\
      --loss_flag 2\
      --itr 1 --batch_size 128 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log
done
done