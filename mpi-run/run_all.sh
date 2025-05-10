#!/bin/bash

REPEATS=10

PROCESSES=(1 2 4 8 16)

EXECUTABLE=exp_mpi

SOURCE_FILE="${EXECUTABLE}.c"

module load mpi/openmpi-x86_64

echo "Компиляция $SOURCE_FILE ..."
mpicc -o $EXECUTABLE $SOURCE_FILE -lm -O3 || {
    echo "Ошибка компиляции $SOURCE_FILE"
    exit 1
}

for procs in "${PROCESSES[@]}"; do
    for ((i = 1; i <= REPEATS; i++)); do
        JOB_NAME="ExpMPI_${procs}_procs_run${i}"

        if [ "$procs" -le 2 ]; then
            WALLTIME="00:05"
            MEM="512MB"
        elif [ "$procs" -le 8 ]; then
            WALLTIME="00:10"
            MEM="1GB"
        else
            WALLTIME="00:20"
            MEM="2GB"
        fi

        bsub <<EOF
#!/bin/bash
#BSUB -J $JOB_NAME
#BSUB -W $WALLTIME
#BSUB -n $procs
#BSUB -R "span[ptile=$procs]"
#BSUB -o logs/output_${JOB_NAME}_%J.out
#BSUB -e logs/error_${JOB_NAME}_%J.err
#BSUB -M $MEM

module load mpi/openmpi-x86_64
mpirun --bind-to core --map-by core ./$EXECUTABLE
EOF

        sleep 0.2
    done
done