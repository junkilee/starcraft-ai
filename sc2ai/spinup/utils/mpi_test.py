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
    n_epochs = 1000
    n_examples = 50
    n_in = 30
    n_out = 3
    write_freq = 50
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
        loss = torch.square(torch.sum(torch.tensor(out_mat, dtype=torch.float32) - mod_out))
        loss.backward()
        mpi_avg_grads(mod)
        print("{}:{} ====================".format(i, proc_id()), mod.mlp.weight.grad)
        optimizer.step()
        if i % write_freq == 0:
            f.write("{}, {}\n".format(i, loss.float()))
        print(datetime.datetime.now(), proc_id(), loss)
    print('ended {}'.format(i))
    torch.save(mod, "{}-module.pt".format(proc_id()))
    f.close()


def mpi_avg_test():
    for i in range(5):
        r = np.random.randint(0, 5)
        a = mpi_avg(r)
        print("i:{}, id:{}, a:{}".format(i, proc_id(), a))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=5)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    mpi_test()
    #mpi_finalize()
