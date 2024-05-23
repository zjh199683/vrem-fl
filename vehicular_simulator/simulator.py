import argparse
import warnings
import pandas as pd
from tqdm import tqdm
import traci
import sumolib

warnings.filterwarnings("ignore")
map_ids_filename = '../scenarios/big_map/new_plain.edg'

parser = argparse.ArgumentParser()
parser.add_argument('--horizon', help='Number of simulation steps', type=int, default=3600)
parser.add_argument('--input_net', help='Path to the input graph (file .net.xml)', type=str, default='padova.net.xml')
parser.add_argument('--cfg_file', help='Path to the configuration file (file .sumocfg)', type=str,
                    default='padova.sumocfg')
parser.add_argument('--output_file', help='Path to the results (file .csv)', type=str, default='veh_info.csv')
args = parser.parse_args()


horizon = args.horizon
net = sumolib.net.readNet(args.input_net)
# NOTE: the simulation step size and the number of cars are to be defined in the .sumocfg
traci.start(["sumo", "-c", args.cfg_file, "--time-to-teleport", "-1"])
info = dict(time=[], veh_ID=[], x=[], y=[])

for step in tqdm(range(horizon)):
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    for veh in vehicles:
        info['time'].append(step)
        info['veh_ID'].append(veh)
        pos = traci.vehicle.getPosition(veh)
        info['x'].append(pos[0])
        info['y'].append(pos[1])

df = pd.DataFrame(info)
df.to_csv(args.output_file)