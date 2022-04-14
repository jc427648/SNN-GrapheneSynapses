#!/bin/bash
for i in {1..99}
do
  qsub Optimization_10.sh
  # qsub Optimization_30.sh
  # qsub Optimization_100.sh
  # qsub Optimization_300.sh
  # qsub Optimization_500.sh
  sleep 5
done
