
accelerate launch\
 train_pair_pos_neg.py\
 --training_file training_file.parquet\
 --training_mode visual_upper_layers_textual_upper_layers\
 --batch_size 32\
 --n_epochs 10\
 --save_folder ./clip_model\
 --loss_name triplet

