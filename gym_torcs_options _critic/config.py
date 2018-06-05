import argparse

parser = argparse.ArgumentParser()




parser.add_argument('--nruns', help="Number of runs", type=int, default=100)


parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", action='store_true', default=True)
parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-2)
parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')
parser.add_argument('--priority', help="Priority of discount ratio between keep and terminate an option", type=float, default=1.0)

parser.add_argument('--vision', help="If activate vision input of torcs", type=bool, default=False)
parser.add_argument('--output_graph', help="Output graph of tensorflow", type=bool, default=True)
parser.add_argument('--option_size', help='Number of options', type=int, default=2)
parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1000)
parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=500)
parser.add_argument('--discount', help='Discount factor', type=float, default=0.999)
parser.add_argument('--epsilon', help="Factor on randomness start point", type=float, default=1.0)
parser.add_argument('--epsilon_decay', help="Decay on randomness", type=float, default=0.995)
parser.add_argument('--epsilon_min', help="Minimal epsilon", type=float, default=0.05)
parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=0.25)
parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=0.25)
parser.add_argument('--lr_critic', help="Learning rate", type=float, default=0.5)

parser.add_argument('--state_size', help="Number of states as input", type=int, default=65)
parser.add_argument('--action_size', help="Number of primitive actions as output", type=int, default=2)

parser.add_argument('--learning_rate_critic', help="learning_rate_critic", type=float, default=0.001)
parser.add_argument('--learning_rate_termination', help="learning_rate_termination", type=float, default=0.0005)

parser.add_argument('--batch_size', help="batch update size", type=int, default=32)
parser.add_argument('--buffer_size', help="batch update size", type=int, default=100000)
parser.add_argument('--learning_rate_actor', help="learning_rate_actor", type=float, default=0.0001)
parser.add_argument('--tau', help="Target Network HyperParameters", type=float, default=0.001)


args = parser.parse_args()
fname = '-'.join(['{}_{}'.format(param, val) for param, val in sorted(vars(args).items())])
fname = 'optioncritic-' + fname + '.npy'
