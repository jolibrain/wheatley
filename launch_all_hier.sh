#!/bin/bash


#for s in 5 10 20 30
for s in 5 10 20 
do
    #for l in 1 2 3 4 5 6
    for l in 7 8 9 10
    do
        echo SIZE $s NLAYERS $l >> RES_HIER_NOUPNODES_10
        python -m pp.bench.solve  --device cuda:3  --size $s   --path /data1/infantes/networks/bench/pp_BENCH_HIER_SIMPLERAGGR_UPCLUSTER_NOUPDATENODES_10 --nlayers $l >> RES_HIER_NOUPNODES_10
        echo SIZE $s NLAYERS $l >> RES_HIER_NOUPNODES_10
    done
done
