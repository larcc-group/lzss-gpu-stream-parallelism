#!/bin/bash
# ./benchmark.py ../data/input.txt 5 >> stdout.out 2>> stderr.out
# ./benchmark.py ../data/input.txt 5  >> stdout.out 2>> stderr.out

./benchmark.py ../data/CreditBackup80.BAK 5 >> stdout.out 2>> stderr.out
./benchmark.py ../data/silesia.tar 5  >> stdout.out 2>> stderr.out
./benchmark.py ../data/linux-4.16-rc4.tar 5  >> stdout.out 2>> stderr.out