import argparse
from sc2ai.spinup.algorithms.ppo.ppo import ppo
from sc2ai.spinup.algorithms.ppo.sc2_nets import SC2AtariNetActorCritic
from sc2ai.spinup.utils.mpi_tools import mpi_fork
from sc2ai.envs import make_sc2env
from absl import flags
import torch


if __name__ == '__main__':
    flags.FLAGS.mark_as_parsed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', type=str, default='FleeRoachesv4_training')
    parser.add_argument('--seed', '-s', type=int, default=0)
    #parser.add_argument('--cpu', type=int, default=1)
    #parser.add_argument('--steps', type=int, default=6000)
    #parser.add_argument('--epochs', type=int, default=1000000)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='ppo_sc2')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from sc2ai.spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("device - ", dev, device)

    ppo(lambda: make_sc2env(map=args.map_name), actor_critic=SC2AtariNetActorCritic,
        ac_kwargs=dict(), # hidden_sizes=[args.hid]*args.l
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=device)