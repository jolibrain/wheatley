#!/bin/sh

# Must provide an output directory as an argument.
if [ $# -ne 1 ]; then
    echo "Usage: $0 [DIR]"
    exit 1
fi

OUT_DIR=$1
echo "Specified output directory: $OUT_DIR."
mkdir -p "$OUT_DIR"

echo "Running pytests."
for test_file in $(find tests/ -type f -name "*.py" ! -name "*_perf.py"); do
    # Run the test and save the output to a file.
    echo "Running $test_file."
    python3 -m pytest --rootdir $OUT_DIR  "$test_file" "$OUT_DIR"

    if [ $? -ne 0 ]; then
        echo "Tests in $test_file failed."
        exit 1
    fi
done

echo "All tests passed."
echo "Deleting output directory."
rm -rf "$OUT_DIR"
