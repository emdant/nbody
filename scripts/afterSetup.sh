#!/bin/bash

sudo cp /home/ubuntu/hostfile ~
sudo cp /home/ubuntu/nbodyV{4..5}.c ~

mpicc -O3 nbodyV4.c -o v4.out -lm
mpicc -O3 nbodyV5.c -o v5.out -lm

hosts="hostfile"

i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $hosts

i=1
for ip in ${ips[@]}; do
  scp ./v4.out pcpc@${ip}:
  scp ./v5.out pcpc@${ip}:
  i=$((i + 1))
done