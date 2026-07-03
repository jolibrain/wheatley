#!/bin/bash

#for l in 10 20 30 40 50
for l in 60 70 80 90 100 110 120 130 140 150
do
    echo NLAYERS $l  >> RES_HARD_chrono_10
    python -m pp.bench.solve --device cuda:1 --hard --size 30 --nlayers $l  --path /data1/infantes/networks/bench/pp_BENCH_chrono_10_g2_VISITEDATTR_SHARED_POOLALL_AFTERCONV_ONL0 >> RES_HARD_chrono_10
    echo NLAYERS $l  >> RES_HARD_chrono_10
done
