#!/bin/sh

# ami-id: ami-09e67e426f25ce0d7
# instance type: m4.large
# security-group: sg-0817026826bf4d911
aws ec2 run-instances --image-id ami-09e67e426f25ce0d7 --security-group-ids sg-0817026826bf4d911 --count 8 --instance-type m4.large --key-name devenv-key
aws ec2 describe-instances --filters "Name=instance-state-name,Values=pending,running" --query "Reservations[*].Instances[*].[PublicIpAddress]" --output=text > hostfile
sed -e 's/$/ slots=2/' -i hostfile

# Copy devenv-key, hostfile, source_file
master=$(head -1 hostfile  | awk '{print $1;}')

scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i devenv-key.pem ./hostfile ./devenv-key.pem ./setup.sh ./generateInstall.sh ./afterSetup.sh ubuntu@$master:
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i devenv-key.pem ../v4/nbodyV4.c ../v5/nbodyV5.c ubuntu@$master:
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i devenv-key.pem ubuntu@$master
