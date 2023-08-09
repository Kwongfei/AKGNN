import os

# Cora
weight_decay = [5e-6, 5e-4, 5e-3, 5e-1]
lr = [0.01]
dropout = [0.2, 0.5, 0.8]

for dp in dropout:
    for wd in weight_decay:
        for learning_rate in lr:
                    os.system('python train_citeseer.py --lr {0} --weight_decay {1} '
                              '--device 2 --dropout {2}'.format(wd, learning_rate, dp))
