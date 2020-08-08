source /home/users/hlli/anaconda3/bin/activate jigsaw

python train_unseen.py \
 --K=1 --T_min=10 --T_max=5 \
 --gamma 2.0 --adv_learning_rate 1.0 \
 --learning_rate 1e-4 \
 --limit_source=10000 \
 --limit_target=10000 \
 --batch_size 30 \
 --epochs 32 \
 --network resnet18 \
 --n_classes 7  --val_size 0.1 \
 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 \
 --source $1 \
 --target ALL \
 --adam \
 --image_size 222 \
 --gpu $2