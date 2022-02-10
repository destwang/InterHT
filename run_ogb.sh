
python run_ogb.py --anchors 20000 --max_paths 20 --do_train -n 128 -adv --use_dists --model InterHT --print_on_screen --cuda -lr 0.0005 --valid_steps 20000 --log_steps 20000 --max_steps 300000 --save_checkpoint_steps 1000000 --do_valid --inverses --test_log_steps 20000 --gamma 6.0 --sample_rels 12 -randomSeed 0 --val_inverses --do_test --double_entity_embedding -tr -a 2 --drop 0.05
