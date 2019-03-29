#!/bin/sh

REPS=2
LINUX="linux"
SILESIA="silesia"
NOW=`date`
eval "mv benchmark.db \"benchmark_${NOW}.db\""
eval "mv benchmark.log \"benchmark_${NOW}.log\""

eval "python3 benchmark.py $LINUX $REPS"
eval "python3 benchmark.py $SILESIA $REPS"