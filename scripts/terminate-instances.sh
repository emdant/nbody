#!/bin/sh

instances=$(aws ec2 describe-instances --query "Reservations[].Instances[].[InstanceId]" --output text | tr '\n' ' ')
aws ec2 terminate-instances --instance-ids $instances

alarms=$(aws cloudwatch describe-alarms --query "MetricAlarms[].[AlarmName]" --output text | tr '\n' ' ')
aws cloudwatch delete-alarms --alarm-names $alarms