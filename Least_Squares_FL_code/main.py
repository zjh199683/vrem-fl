import pickle
from FL_LS_Algorithms import Fed_GD_LS
from utils import *

# 配置参数
class Config:
    def __init__(self):
        self.filename = '../new_corr_datasetEstim_250_2.csv'
        self.time_slot = 1  # 时间槽长度(秒)
        self.reg_lambda = 1e-4  # 正则化参数
        self.rounds = 30
        self.n = 25  # 模型参数
        self.m = 30  # 每轮最大可调度客户端数
        self.param_dim = 128  # 比特数
        self.noniid = True
        self.aoi = False
        self.save = True
        self.scheduling = 'optimal'
        self.comp = 'opt' 
        self.tx = 'opt'

def load_data(config):
    print('导入车辆数据...')
    data = import_vehicles_data(filename=config.filename, 
                              fields={'bitrate', 'estimBitrate'}, 
                              min_time=600)
    IDs = list(data.keys())
    data = {k: data[k] for k in IDs}
    print('车辆数据导入完成')
    return data, IDs

def generate_dataset(data, config):
    print('生成合成数据集...')
    M = len(data)
    X, Y = getSyntheticDataset(sizeXUser=100, M=M, n=config.n, r=config.n, 
                              sigma=1e-5, s_min=-1, s_max=0, noniid=config.noniid)
    print('数据集生成完成')
    
    X, Y, Ds, Ys = setFL_DS_LSs(X, Y, M)
    return X, Y, Ds, Ys, M

def main():
    config = Config()
    
    # 加载数据
    data, IDs = load_data(config)
    X, Y, Ds, Ys, M = generate_dataset(data, config)
    
    # 准备训练数据
    Ds = {IDs[k]: Ds[k] for k in range(M)}
    Ys = {IDs[k]: Ys[k] for k in range(M)}
    
    # 训练参数设置
    max_latency = 100 * config.time_slot
    model_size = config.n * config.param_dim
    
    # 执行联邦学习
    costs, distancesFromOpt, slots, steps, tx_steps = Fed_GD_LS(
        X, Y, Ds, Ys, model_size, config.rounds, data, config.time_slot, 
        config.m, max_latency, mobility=True, comp=config.comp, 
        scheduling=config.scheduling, batch_size=1, comp_slots_min=1, 
        reg_lambda=config.reg_lambda, tx=config.tx, aoi_only=config.aoi, beta=0
    )
    
    # 保存结果
    if config.save:
        save_results(slots, distancesFromOpt, steps, tx_steps, 
                    config.noniid, config.aoi, config.scheduling, 
                    config.comp, config.tx)

def save_results(slots, distancesFromOpt, steps, tx_steps, noniid, aoi, 
                scheduling, comp, tx):
    results = dict(
        time=slots,
        convergence=distancesFromOpt,
        avgCompSteps=steps,
        avgTxSteps=tx_steps
    )
    
    suffix = '_noniid' if noniid else ''
    suffix += '_aoi' if aoi else ''
    
    filename = f'../estBitrate_250_{scheduling}_{comp}_{tx}{suffix}_beta0.pk'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main()