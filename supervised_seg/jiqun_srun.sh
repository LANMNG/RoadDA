srun -A test -J supervised_easy -N 1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 -p gpu -t 0-00:40:00 python test.py
