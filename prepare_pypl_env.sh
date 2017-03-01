# This needs to be sourced in order for the ProBT-based modules to work.
PL_DIR="/home/vsantos/Desktop/probt-spl-3.0.0-linux64-dynamic-release/lib/"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PL_DIR}

export PYTHONPATH=${PL_DIR}:${PYTHONPATH} "$@"