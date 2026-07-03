#!/bin/bash


for s in 5 10 20 30
do
    for l in 4 8 12 16 20 24 28 
    do
        echo SIZE $s NLAYERS $l >> RES_CHRONO_H32_10
        python -m pp.bench.solve --device cuda:0  --size $s   --path /data1/infantes/networks/bench/pp_BENCH_chrono_H32_10 --nlayers $l >> RES_CHRONO_H32_10
        echo SIZE $s NLAYERS $l >> RES_CHRONO_H32_10
    done
done
