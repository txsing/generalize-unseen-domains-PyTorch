source /home/users/hlli/anaconda3/bin/activate jigsaw

python train_unseen.py \
 --K=0 --T_min=100 --T_max=15 \
 --limit_source=10000 \
 --limit_target=10000 \
 --gamma 1.0 --adv_learning_rate 1.0 \
 --batch_size 32 \
 --epochs 32 \
 --network lenet \
 --n_classes 10 --learning_rate 1e-4 --val_size 0.1 \
 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 \
 --source mnist \
 --target ALL \
 --image_size 32 \
 --adam \
 --flip_p 0.5 \
 --gpu $1