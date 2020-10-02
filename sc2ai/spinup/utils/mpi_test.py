import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from sc2ai.spinup.utils.mpi_pytorch import mpi_avg_grads
from sc2ai.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, num_procs, mpi_finalize


class TestModule(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.mlp = nn.Linear(n_in, n_out)

    def forward(self, inp):
        return self.mlp(inp)


def mpi_test():
    n_epochs = 500
    n_examples = 30
    n_in = 30
    n_out = 3
    # generate baseline
    W = np.random.rand(n_out, n_in)
    mod = TestModule(n_in, n_out)
    optimizer = Adam(mod.parameters(), lr=1e-3)

    f = open("{}-output.txt".format(proc_id()), "w")
    for i in range(n_epochs):
        # generate examples
        inp_mat = np.zeros((n_examples, n_in))
        out_mat = np.zeros((n_examples, n_out))
        for j in range(n_examples):
            inp_mat[j] = np.random.rand(n_in)
            out_mat[j] = np.matmul(W, inp_mat[j].reshape(-1, 1)).flatten()
        mod_out = mod(torch.tensor(inp_mat, dtype=torch.float32))
        optimizer.zero_grad()
        loss = torch.sqrt(torch.sum(torch.tensor(out_mat, dtype=torch.float32) - mod_out))
        loss.backward()
        mpi_avg_grads(mod)
        optimizer.step()
        f.write("{}, {}\n".format(i, loss.float()))
        print(datetime.datetime.now(), proc_id(), loss)
    print('ended {}'.format(i))
    f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=5)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    mpi_test()
    #mpi_finalize()
