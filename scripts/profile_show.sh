THRESHOLD=$1
if [ -z "$THRESHOLD" ]
then
	THRESHOLD=5
fi
gprof2dot -f pstats profile.data -n $THRESHOLD | dot -Tpdf > profile.pdf
xdg-open profile.pdf
