
accelerate launch --config_file accelerate_fp16.yaml run_ogb_plus.py --do_train -b 512 -n 64 -adv --model InterHT --print_on_screen --cuda -lr 0.0005 --valid_steps 60000 --log_steps 60000 --max_steps 500000 --save_checkpoint_steps 1000000 --do_valid --inverses --test_log_steps 60000 --gamma 6.0 -randomSeed 0 --val_inverses --do_test -a 2 --drop 0.05 -d 256

