source /home/users/hlli/anaconda3/bin/activate jigsaw
pacs=('photo' 'art_painting' 'cartoon' 'sketch')

if [ $1 == "caffenet" ]; then
  imgsize=225
  hflip=0.0
  jitter=0.0
else # resnet18
  imgsize=222
  hflip=0.5
  jitter=0.4
fi

for sd in ${pacs[@]}; do
    echo source-${sd}
    python train_unseen.py --K=1 --T_min=10 --T_max=5 --gamma 1.0 --adv_learning_rate 2.0 --batch_size 32 --epochs 30 --network $1 --n_classes 7 --learning_rate 1e-4 --val_size 0.1 --min_scale 0.8 --max_scale 1.0 --random_horiz_flip ${hflip} --jitter ${jitter} --source ${sd} --target ALL --adam --image_size ${imgsize} --gpu $2
    echo END
done