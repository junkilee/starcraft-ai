import multiprocessing
import numpy as np
import os
import torch
from mpi4py import MPI
from sc2ai.spinup.utils.mpi_tools import broadcast, mpi_avg, num_procs, proc_id


def setup_pytorch_for_mpi():
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)


def mpi_avg_grads(module):
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.cpu().numpy()
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    if num_procs() == 1:
        return
    for p in module.parameters():
        p_numpy = p.data.cpu().numpy()
        broadcast(p_numpy)