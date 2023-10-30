#/bin/bash

start_bid=$1
k=$2
cuda_visible_device=$3

# run inference for k bidxes starting from start_bid
set -e
for ((i=0; i<$k; i++))
do
    bid=$((start_bid+i))
    echo "Running inference for bid $bid"
    CUDA_VISIBLE_DEVICES="${cuda_visible_device}" python -u llama2oie.py --machine gala1sgl --model_root ../lms --model_name llama2-7b-chat-hf \
    --task predict --data_root ./newsspike/splits --inference_id "${bid}" --peft_type lora --pad_method bos \
    --lr 5e-5 --lora_r 64 --eval_bsz 8 --debug --use_16_full
done