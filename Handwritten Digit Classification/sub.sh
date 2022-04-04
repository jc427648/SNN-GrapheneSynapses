#!/bin/bash
for i in {1..124}
do
  qsub Optimization_10.sh
  sleep 10
done
