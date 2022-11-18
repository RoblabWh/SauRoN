#!/bin/bash

ipadress=$(hostname -I | cut -d' ' -f1)
number=${ipadress: -3}

case $number in
    121)
       echo -n "121"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 10 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    120)
       echo -n "120"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 100 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    119)
       echo -n "119"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 10 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    118)
       echo -n "118"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 100 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    101)
       echo -n "101"
       args="--ckpt_folder ./models/$number --level_files Funnel.svg --update_experience 100000 --batch_size 10 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    116)
       echo -n "116"
       args="--ckpt_folder ./models/$number --level_files Funnel.svg --update_experience 100000 --batch_size 100 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    115)
       echo -n "115"
       args="--ckpt_folder ./models/$number --level_files Funnel.svg --update_experience 100000 --batch_size 10 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    114)
       echo -n "114"
       args="--ckpt_folder ./models/$number --level_files Funnel.svg --update_experience 100000 --batch_size 100 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    113)
       echo -n "113"
       args="--ckpt_folder ./models/$number --level_files SimpleObstacles.svg --update_experience 100000 --batch_size 10 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    112)
       echo -n "112"
       args="--ckpt_folder ./models/$number --level_files SimpleObstacles.svg --update_experience 100000 --batch_size 100 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    111)
       echo -n "111"
       args="--ckpt_folder ./models/$number --level_files SimpleObstacles.svg --update_experience 100000 --batch_size 10 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    110)
       echo -n "110"
       args="--ckpt_folder ./models/$number --level_files SimpleObstacles.svg --update_experience 100000 --batch_size 100 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    109)
       echo -n "109"
       args="--ckpt_folder ./models/$number --level_files Simple_12.svg --update_experience 100000 --batch_size 10 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    108)
       echo -n "108"
       args="--ckpt_folder ./models/$number --level_files Simple_12.svg --update_experience 100000 --batch_size 100 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    107)
       echo -n "107"
       args="--ckpt_folder ./models/$number --level_files Simple_12.svg --update_experience 100000 --batch_size 10 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    106)
       echo -n "106"
       args="--ckpt_folder ./models/$number --level_files Simple_12.svg --update_experience 100000 --batch_size 100 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    105)
       echo -n "105"
       args="--ckpt_folder ./models/$number --level_files SwapSide_a.svg --update_experience 100000 --batch_size 10 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    104)
       echo -n "104"
       args="--ckpt_folder ./models/$number --level_files SwapSide_a.svg --update_experience 100000 --batch_size 100 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    103)
       echo -n "103"
       args="--ckpt_folder ./models/$number --level_files SwapSide_a.svg --update_experience 100000 --batch_size 10 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    102)
       echo -n "102"
       args="--ckpt_folder ./models/$number --level_files SwapSide_a.svg --update_experience 100000 --batch_size 100 --K_epochs 40 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    117)
       echo -n "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 10 --K_epochs 42 --log_interval 5 --steps 2000 --gamma 0.99"
       echo -n "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
       ;;
    *)
       echo -n "du dumm"
       ;;
esac


cd ~/Dokumente/SauRoN/
mkdir ./models/$number
source "/home/nex/anaconda3/etc/profile.d/conda.sh"
export PATH="/home/nex/anaconda3/bin:$PATH"
conda activate sauron
echo "$args" > ./models/$number/args.txt

python main.py $args
