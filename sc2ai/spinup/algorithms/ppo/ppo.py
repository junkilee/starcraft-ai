import sys
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
import gym
import time
import sc2ai.spinup.algorithms.ppo.core as core
import sc2ai.spinup.algorithms.ppo.sc2_nets as sc2_nets
from sc2ai.spinup.utils.logx import EpochLogger
from sc2ai.spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from sc2ai.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs




class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device=torch.device('cpu')):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in data.items()}


def ppo(env_fn, actor_critic=sc2_nets.SC2AtariNetActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=10000,
        epochs=1000000, gamma=0.99, clip_ratio=0.2, lr=3e-4, vf_coeff=0.5, ent_coeff=0.01, train_iters=10, lam=0.97,
        max_ep_len=1000, target_kl=0.03, batch_size=64, logger_kwargs=dict(), save_freq=100, device=torch.device("cpu")):
    setup_pytorch_for_mpi()

    print("device - ", device)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    # Later change this ---- This depends on the environment and the network structure
    obs_space = env.observation_gym_space
    obs_dim = obs_space['feature_screen'].shape
    act_dim = env.action_gym_space.nvec.shape
    # ----------------------
    print("obs_dim, act_dim = ", obs_dim, act_dim)

    action_spec, action_mask = env.action_set.get_action_spec_and_action_mask()
    ac = actor_critic(env.observation_gym_space,
                      action_spec=action_spec, action_mask=action_mask, device=device, **ac_kwargs)

    sync_params(ac)

    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam, device)

    def compute_loss_pi(data, start, end):
        obs, act, adv, logp_old = data['obs'][start:end], data['act'][start:end], \
                                  data['adv'][start:end], data['logp'][start:end]

        pis, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = 0
        for pi in pis:
            if isinstance(pi, tuple):
                ent += pi[0].entropy().mean() + pi[1].entropy().mean()
            else:
                ent += pi.entropy().mean()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32, device=device).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent.item(), cf=clipfrac)

        return loss_pi, ent, pi_info

    def compute_loss_v(data, start, end):
        obs, ret = data['obs'][start:end], data['ret'][start:end]
        return ((ac.v(obs) - ret) ** 2).mean()

    optimizer = Adam(ac.parameters(), lr=lr)
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, ent_old, pi_info_old = compute_loss_pi(data, 0, len(data['obs']))
        pi_l_old = pi_l_old.item()
        ent_old = ent_old.item()
        v_l_old = compute_loss_v(data, 0, len(data['obs'])).item()

        # Change this part to combined batch training

        for i in range(train_iters):
            # do mini batch training instead of full batch
            print("training pi and v ...")
            sys.stdout.flush()
            data_length = len(data['obs'])
            for j in range(data_length // batch_size):
                print("doing batch {}".format(j))
                sys.stdout.flush()
                start = batch_size * j
                end = batch_size * (j + 1)
                if end > data_length:
                    end = data_length
                optimizer.zero_grad()
                loss_pi, entropy, pi_info = compute_loss_pi(data, start, end)
                loss_v = compute_loss_v(data, start, end)
                #kl = mpi_avg(pi_info['kl'])
                #if kl > 1.5 * target_kl:
                #    logger.log('Early stopping at step %d due to reaching max kl.' % i)
                #    break
                (loss_pi + vf_coeff * loss_v - ent_coeff * entropy).backward()
                mpi_avg_grads(ac)
                optimizer.step()

        logger.store(StopIter=i)
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old), DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            o = o['feature_screen']
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0))

            #print("a v logp -- ", a, v, logp)
            #print(o.shape)

            #for i in range(84):
            #    for j in range(84):
            #        print("{:3.1f} ".format((o[3, i, j])), end='')
            #    print()

            print(".", end='')
            sys.stdout.flush()

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            #print(a, r, v, logp)

            buf.store(o, a, r, v, logp)
            logger.store(VVals=v, incre=True)

            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                print("episode ended {}".format(t))
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                if timeout or epoch_ended:
                    o = o['feature_screen']
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32).to(device).unsqueeze(0))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, epoch)

        print("update started....")
        sys.stdout.flush()
        update()
        print("update ended....")
        sys.stdout.flush()

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from sc2ai.spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

