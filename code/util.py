import argparse
import logging
import os
import pdb
import random
import subprocess
from collections import namedtuple, defaultdict
from os.path import join

import numpy as np
import scipy.sparse as sparse
import torch
import yaml

logger = logging.getLogger(__name__)

DataSample = namedtuple('DataSample', ['filename', 'formula', 'adj', 'sat'])


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, default=0)  # default=None
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        try:
            config_data = f.read()
        except IOError:
            print("Can't read file: ", args.config_path)
    config = yaml.load(config_data)
    config['dir'] = join('results', config['name'])
    os.makedirs(config['dir'], exist_ok=True)

    log_file = join(config['dir'], 'train.log')
    logging.basicConfig(
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()],
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    logger.info('Configuration:\n' + config_data)

    if config['seed']:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    use_gpu = args.gpu is not None and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_gpu else 'cpu')
    logger.info(f'Device: {device}')

    config['no_eval'] = not config['eval_set']

    config = defaultdict(lambda: None, config)
    return config, device


def adj_sign(n, m, occur):
    i = np.repeat(range(n), [len(lst) for lst in occur])
    j = np.concatenate(occur)
    v = np.ones(len(i), dtype=np.int64)
    return sparse.coo_matrix((v, (i, j)), shape=(n, m))


def adj(f):
    n, m, occur = f.n_variables, len(f.clauses), f.occur_list
    adj_pos = adj_sign(n, m, occur[1: n + 1])
    adj_neg = adj_sign(n, m, occur[:n:-1])
    return adj_pos, adj_neg


def adj_batch(adjs, fstack):
    adjp, adjn = list(zip(*adjs))
    return fstack((sparse.block_diag(adjp), sparse.block_diag(adjn)))


def to_sparse_tensor(x):
    x = x.tocoo()
    i = torch.tensor(np.vstack((x.row, x.col)), dtype=torch.int64)
    v = torch.tensor(x.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, torch.Size(x.shape))


def init_edge_attr(k):
    return torch.cat(
        (
            torch.tensor([1, 0], dtype=torch.float32).expand(k, 2),
            torch.tensor([0, 1], dtype=torch.float32).expand(k, 2),
        ),
        dim=0,
    )


def normalize(x):
    return 2 * x - 1


def unnormalize(x):
    return (x + 1) / 2


def get_arg_parser(name):
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='output directory')
    parser.add_argument('N', type=int, help='number of problems to be generated')
    parser.add_argument('n', type=int, help='number of nodes')
    parser.add_argument('p', type=float, help='probability of edge')
    parser.add_argument('id', type=int, help='starting id')
    parser.add_argument('k', type=int, help='size of the {}'.format(name))
    return parser


def create_sat_problem(filename, n, p, k, sat_name=None):
    while True:
        subprocess.call(['cnfgen', '-q', '-o', 'tmp.cnf', sat_name, '--gnp', str(n), str(p), str(k)])
        try:
            subprocess.check_call(['minisat', 'tmp.cnf'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            if ex.returncode == 10:
                os.rename('tmp.cnf', filename)
                return
            os.remove('tmp.cnf')


def process_sat(sat, name):
    parser = get_arg_parser(name=name)
    args = parser.parse_args()

    try:
        os.makedirs(args.dir)
    except OSError:
        if not os.path.isdir(args.dir):
            raise

    os.chdir(args.dir)

    for i in range(args.N):
        filename = 'id={}_n={}_p={}_k={}.cnf'.format(args.id + i, args.n, args.p, args.k)
        create_sat_problem(filename, args.n, args.p, args.k, sat_name=sat)
