#!/bin/bash


#for s in 5 10 20 30
for s in 30
do
#    for l in 4 8 12 16 20 24 28
    for l in 8 12 16 20 24 28             
    do
        echo SIZE $s NLAYERS $l >> RES_WPR_10
        python -m pp.bench.solve --device cuda:2  --size $s   --path /data1/infantes/networks/bench/pp_BENCH_WPR_10 --nlayers $l >> RES_WPR_10
        echo SIZE $s NLAYERS $l >> RES_WPR_10
    done
done
