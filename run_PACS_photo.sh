python train_unseen.py \
 --K=6 --T_min=10 --T_max=5 \
 --gamma 1.0 --adv_learning_rate 2.0 \
 --batch_size 32 \
 --epochs 30 \
 --network caffenet \
 --n_classes 7 --learning_rate 0.001 --val_size 0.1 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 \
 --source photo \
 --target cartoon \
 --image_size 222
