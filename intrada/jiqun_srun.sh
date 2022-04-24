srun -A test -J intrada_test -N 1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 -p gpu -t 3-00:00:00 python test.py
