from math import fabs
from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import pickle
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
def store_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # get the params
    args = get_args()
    time_scale = 1 if args.evaluate == True else 40
    env, args = make_env(args,"/home/arcyl/new/small map return flag/small_map_return_flag.x86_64", time_scale)
    runner = Runner(args, env)
    train=args.evaluate==False
    if not train:
        runner.evaluate()
        runner.plot_graph(runner.avg_returns_test,method='test')
    else:
        print("here")
        train=runner.run()
        store_data(args.save_dir + '/' + args.scenario_name +'/train_purple.txt',train['team_purple'])
        store_data(args.save_dir + '/' + args.scenario_name +'/train_blue.txt',train['team_blue'])
