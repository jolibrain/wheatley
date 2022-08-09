bench() {
	JOBS=$1
	TASKS=$2
	ARGS=$3
	COMMAND="python3 train.py
		--n_j $JOBS --n_m $TASKS
		--n_steps_episode 1024 --total_timesteps 1024 --validation_freq 99999
		--duration_type stochastic --fixed_problem --features duration
		--reward_model_config optimistic --ortools_strategy averagistic
		--batch_size 32 --max_edges_upper_bound_factor 2 --device $DEVICE
		$ARGS"
	
	rm -f *.tmp
	START=$(date +%s)
	$COMMAND > fps.tmp &
	PID=$!

	while true
	do
		gpustat -cpu >> gpu.tmp
		ps -p $PID -o rss= >> mem.tmp
		sleep 1
	done &
	LOOP=$!

	echo $COMMAND
	wait $PID
	kill $LOOP

	STOP=$(date +%s)
	TIME=$(($STOP - $START))
	FPS=$(fgrep fps fps.tmp | tail -1 | egrep -o '[0-9]+')
	MEM=$(($(sort -g mem.tmp | tail -1) / 1024))
	GPU=$(egrep -o "python3/$PID\([0-9]+M\)" gpu.tmp | cut -f2 -d'(' | cut -f1 -dM | sort -g | tail -1)

	echo "$JOBS	$TASKS	$DEVICE	$ARGS	$TIME	$FPS	$MEM	$GPU" >> benchmark/perf.csv
}

echo "JOBS	TASKS	DEVICE	ARGS	TIME	FPS	MEM	GPU" > benchmark/perf.csv

for DEVICE in cpu cuda:0
do
	for SIZE in 4 8 12 16 20 24 28 32
	do
		bench $SIZE $SIZE
		test $SIZE -eq 16 && continue
		bench $SIZE 16
		bench 16 $SIZE
	done
	bench 12 16 "--load_problem instances/agilea/small_12_unc.txt"
done

# this one works only on GPU
DEVICE="cuda:0"
bench 200 16 "--load_problem instances/agilea/3B-OF_en_cours_ou_dispo_16_colonnes.txt --load_max_jobs 200 --generate_duration_bounds 0.05"
