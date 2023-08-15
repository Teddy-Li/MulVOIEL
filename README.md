## Scripts

- ` srun -p gpu --gres gpu:2 --kill-on-bad-exit --qos gpu --pty bash -i `
- ` CUDA_VISIBLE_DEVICES="3,4" python llama2_playground.py --model_root ../lms --model_name llama2-7b-chat-hf --task examples --machine gala1`
 
If more than one node is required, exclusive mode --exclusive and --gres=gpu:4 options must be included in your submission script.