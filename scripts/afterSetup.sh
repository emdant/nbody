#!/bin/bash

sudo cp /home/ubuntu/hostfile ~
sudo cp /home/ubuntu/nbody.c ~

mpicc -O3 nbody.c -o nbody.out -lm

hosts="hostfile"

i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $hosts

i=1
for ip in ${ips[@]}; do
  scp ./nbody.out pcpc@${ip}:
  i=$((i + 1))
done

header="NPROC;TIME(s)-BY-NODE;TIME(s)-BY-SLOT;NBODIES;NITER"
# Weak scaling
# mpirun -hostfile hostfile --map-by node/slot -np i nbody.out 100 1250*i
n=1000
echo $header > nbody_weak.csv
for i in {1..16};
do
time_node=$(mpirun -hostfile hostfile --map-by node -np $i v4.out 100 $(($n*$i)) | tail -1 | tr '\n' ';')
time_slot=$(mpirun -hostfile hostfile --map-by slot -np $i v4.out 100 $(($n*$i)) | tail -1 | tr '\n' ';')
echo "$i;${time_node}${time_slot}$(($n*$i));100" >> nbody_weak.csv
done
echo "Done weak scaling"

# Strong scaling
# mpirun -hostfile hostfile --map-by node/slot -np i nbody.out 100 20000
echo $header > nbody_strong.csv
for i in {1..16};
do
time_node=$(mpirun -hostfile hostfile --map-by node -np $i nbody.out 100 16000 | tail -1 | tr '\n' ';')
time_slot=$(mpirun -hostfile hostfile --map-by slot -np $i nbody.out 100 16000 | tail -1 | tr '\n' ';')
echo "$i;${time_node}${time_slot}16000;100" >> nbody_strong.csv
done
echo "Done strong scaling"
