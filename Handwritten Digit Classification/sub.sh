#!/bin/bash
for i in {1..25}
do
  qsub Optimization_10.sh
  sleep 10
done
