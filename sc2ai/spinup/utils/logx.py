import json
import joblib  # https://joblib.readthedocs.io/en/latest/
import shutil
import numpy as np
import torch
import os.path as osp
import time
import atexit
import os
import warnings
from sc2ai.spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar
from sc2ai.spinup.utils.serialization_utils import convert_json

from torch.utils.tensorboard import SummaryWriter


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        if proc_id() == 0:
            self.output_dir = output_dir or "/tmp/experiments/%i" % self.output_dir
            if osp.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway." % self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s" % self.output_file.name, 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you din't include in the first iteration" % key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_config(self, config):
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id() == 0:
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, itr=None):
        if proc_id() == 0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl' % itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log("Warning: could not pickle state_dict", color='red')
            # place holder for tensorflow
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        if proc_id() == 0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = osp.join(self.output_dir, fpath)
            fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
            fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        if proc_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%' + '%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, '__float__') else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()
        self.global_step = 0
        self.global_epoch = 0
        self.tboardwriter = SummaryWriter(self.output_dir)

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'incre':
                self.global_step += 1
                continue
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
            self.tboardwriter.add_scalar('StepProgress/'+str(k), v, self.global_step)
            self.tboardwriter.flush()


    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        if key == 'Epoch':
            self.global_epoch = val
        if val is not None:
            super().log_tabular(key, val)
            self.tboardwriter.add_scalar('EpProgress/'+str(key), val, self.global_epoch)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            self.tboardwriter.add_scalar('EpProgress/'+key if average_only else 'Average' + key, stats[0], self.global_epoch)
            if not average_only:
                super().log_tabular('Std' + key, stats[1])
                self.tboardwriter.add_scalar('EpProgress/'+'Std' + key, stats[1], self.global_epoch)
            if with_min_and_max:
                super().log_tabular('Max' + key, stats[3])
                super().log_tabular('Min' + key, stats[2])
                self.tboardwriter.add_scalar('EpProgress/'+'Min' + key, stats[2], self.global_epoch)
                self.tboardwriter.add_scalar('EpProgress/'+'Max' + key, stats[3], self.global_epoch)
        self.tboardwriter.flush()
        self.epoch_dict[key] = []

    def get_stats(self, key):
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi_statistics_scalar(vals)

    def dump_tabular(self):
        self.tboardwriter.close()
        super().dump_tabular()


