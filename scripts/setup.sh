#!/bin/bash
source ./generateInstall.sh

file="hostfile"
key="devenv-key.pem"
script="install.sh"

i=0
ips[0]=""
while read line; do
  ips[i]=$(grep -o "^[^ ]*" <<< $line)
  i=$((i + 1))
done < $file

i=1
for ip in ${ips[@]}; do
  echo `ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ${key} ubuntu@${ip} "bash -s" < ${script}  &> log${i}.txt &`
  i=$((i + 1))
done
