#!/bin/bash

ipadress=$(hostname -I | cut -d' ' -f1)
number=${ipadress: -3}

case $number in
    116)
       echo -n "116"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 10000 --K_epochs 10 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    115)
       echo -n "115"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 10000 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    114)
       echo -n "114"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 1000 --K_epochs 10 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    112)
       echo -n "112"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 1000 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    107)
       echo -n "107"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 100 --K_epochs 10 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    105)
       echo -n "105"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 100 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    104)
       echo -n "104"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 10 --K_epochs 10 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    103)
       echo -n "103"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 10 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    102)
       echo -n "102"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 1 --K_epochs 10 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    101)
       echo -n "101"
       args="--ckpt_folder ./models/$number --level_files svg3_tareq.svg --update_experience 100000 --batch_size 1 --K_epochs 20 --log_interval 5 --steps 2000 --gamma 0.99"
       ;;
    *)
       echo -n "du dumm"
       ;;
esac


cd ~/Dokumente/SauRoN/
mkdir ./models/$number
echo "$args" > ./models/$number/args.txt

~/anaconda3/envs/sauron/bin/python main.py $args
