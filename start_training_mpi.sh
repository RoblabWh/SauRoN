#!/bin/bash

ipadress=$(hostname -I | cut -d' ' -f1)
number=${ipadress: -3}

case $number in
    120)
       echo -n "120"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 10000 --gamma 0.99 --level_files tunnel.svg Funnel.svg svg3_tareq.svg"
       ;;
    119)
       echo -n "119"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 10000 --gamma 0.99 --level_files tunnel.svg Funnel.svg svg3_tareq.svg"
       ;;
    118)
       echo -n "118"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 10000 --gamma 0.99 --level_files tunnel.svg Funnel.svg"
       ;;
    117)
       echo -n "117"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 10000 --gamma 0.99 --level_files tunnel.svg Funnel.svg"
       ;;
    116)
       echo -n "116"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 2000 --gamma 0.99 --level_files svg3_tareq.svg"
       ;;
    115)
       echo -n "115"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 2000 --gamma 0.99 --level_files svg3_tareq.svg"
       ;;
    114)
       echo -n "114"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 8 --log_interval 5 --steps 2000 --gamma 0.99 --level_files tunnel.svg Funnel.svg svg3_tareq.svg"
       ;;
    113)
       echo -n "113"
       args="--ckpt_folder ./models/$number --mode train --update_experience 256 --sync_experience 256 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 2000 --gamma 0.99 --level_files tunnel.svg Funnel.svg svg3_tareq.svg"
       ;;
    112)
       echo -n "112"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 8 --log_interval 5 --steps 2000 --gamma 0.99 --level_files tunnel.svg Funnel.svg"
       ;;
    173)
       echo -n "173"
       args="--ckpt_folder ./models/$number --mode train --update_experience 256 --sync_experience 256 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 2000 --gamma 0.99 --level_files tunnel.svg Funnel.svg"
       ;;
    110)
       echo -n "110"
       args="--ckpt_folder ./models/$number --mode train --update_experience 128 --sync_experience 128 --batch_size 1 --K_epochs 8 --log_interval 5 --steps 2000 --gamma 0.99 --level_files svg3_tareq.svg"
       ;;
    109)
       echo -n "109"
       args="--ckpt_folder ./models/$number --mode train --update_experience 256 --sync_experience 256 --batch_size 1 --K_epochs 4 --log_interval 5 --steps 2000 --gamma 0.99 --level_files svg3_tareq.svg"
       ;;
    *)
       echo -n "du dumm"
       ;;
esac


cd ~/Dokumente/SauRoN/
mkdir ./models/$number
echo "$args" > ./models/$number/args.txt
source "/home/nex/anaconda3/etc/profile.d/conda.sh"
export PATH="/home/nex/anaconda3/bin:$PATH"
conda activate sauron
mpirun -n 6 python main.py $args
