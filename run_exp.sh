#!/bin/bash

# 设置实验参数
NUM_ROUNDS=200
BATCH_SIZE=32
LEARNING_RATE=1e-4

# HomoLoRA实验
for r in 1 5 20 50; do
    for model_size in xxs xs; do
        for ((round=0; round<$NUM_ROUNDS; round++)); do
            python run_homlora_experiment.py \
            --model_size $model_size \
            --lora_rank $r \
            --round $round \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --output_file homlora_results.json
        done
    done

    if [[ $r -eq 5 || $r -eq 50 ]]; then
        for ((round=0; round<$NUM_ROUNDS; round++)); do
            python run_homlora_experiment.py \
            --model_size reddit \
            --lora_rank $r \
            --round $round \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --output_file homlora_results.json
        done
    fi
done

# HetLoRA实验
for dataset_name in reddit msc; do
    for ((round=0; round<$NUM_ROUNDS; round++)); do
        python run_hetlora_experiment.py \
        --dataset_name $dataset_name \
        --round $round \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --output_file hetlora_results.json
    done
done

# FullFT实验
for dataset_name in reddit msc; do
    for ((round=0; round<$NUM_ROUNDS; round++)); do
        python run_fullft_experiment.py \
        --dataset_name $dataset_name \
        --round $round \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --output_file fullft_results.json
    done
done