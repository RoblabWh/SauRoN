#!/bin/bash

ipadress=$(hostname -I | cut -d' ' -f1)
number=${ipadress: -3}

case $number in
    120)
       echo -n "120"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 100 --K_epochs 10 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
       ;;
    119)
       echo -n "119"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 100 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
       ;;
    118)
       echo -n "118"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 100 --K_epochs 30 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
       ;;
    117)
       echo -n "117"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 100 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
       ;;
    116)
       echo -n "116"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 10 --K_epochs 10 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
       ;;
    115)
       echo -n "115"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 10 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
       ;;
    114)
       echo -n "114"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 10 --K_epochs 30 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
       ;;
    113)
       echo -n "113"
       args="--ckpt_folder ./models/$number --mode train --update_experience 100000 --batch_size 10 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99 --visualization none"
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
