## Scripts

- ` srun -p gpu --gres gpu:2 --kill-on-bad-exit --qos gpu --pty bash -i `
- ` CUDA_VISIBLE_DEVICES="6,7" nohup python -u llama2oie.py --model_root ../lms --model_name llama2-7b-chat-hf --task peft --machine gala1 --lr 5e-5 --pad_method bos --peft_type lora --lora_r 64 --num_epochs 20 > ./nhlogs/llama2oie_7bc_lora64_bos_5e-5.log & ` 

- Evaluate: ` CUDA_VISIBLE_DEVICES="4,5" nohup python -u llama2oie.py --model_root ../lms --model_name llama2-7b-chat-hf --task inference --machine gala1 --peft_type lora --pad_method bos --lr 5e-5 --lora_r 64 --eval_bsz 1 --f_score_beta 0.5 --debug > ./nhlogs/llama2oie_7bc_lora64_bos_5e-5_eval_test.log & `

If more than one node is required, exclusive mode --exclusive and --gres=gpu:4 options must be included in your submission script.

### Transfer Scripts from cirrus to gala1
- ` rsync -a -e "ssh -i ~/.ssh/id_rsa_cirrus" teddy@login.cirrus.ac.uk:/work/ec216/ec216/teddy/llamaOIE ./ `
- ` scp -r -i ~/.ssh/id_rsa_cirrus teddy@login.cirrus.ac.uk:/work/ec216/ec216/shared/lms/llama2-7b-chat-hf ./ `

### Transfer Scripts from gala1 to cirrus
- ` rsync -a -e "ssh -i ~/.ssh/id_rsa_cirrus" ./llamaOIE teddy@login.cirrus.ac.uk:/work/ec216/ec216/teddy/ `

### Bug log
- LLaMA2 generate not compatible with do_sample=True, causes error: `RuntimeError: probability tensor contains either inf, nan or element < 0`
- HuggingFace Trainer bug: when doing trainer.evaluate() with the test set, the trainer gets stuck after batch 23 / 53 (bsz 8), cpu usage stays at 100% with no progress. Problem does not exist when evaluating with the validation set, or doing evaluation manually without the trainer.