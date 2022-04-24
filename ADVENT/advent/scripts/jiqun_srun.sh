srun -A test -J interDA_test -N 1 --ntasks-per-node=1 --cpus-per-task=4 --gres=gpu:1 -p gpu -t 0-00:30:00 python test.py
