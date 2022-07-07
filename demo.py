from fllib.base import *

config = {
    'dataset': {
        'data_name': 'mnist',
        'download': False,
        'distribution_type': 'iid',
        'alpha': 0.5,
        'simulated': True
    },
    'server': {
        'clients_num': 10,
        'rounds': 50,
        'clients_per_round': 5,
        'aggregation_rule': 'fedavg',
        'aggregation_detail': {
            'type': 'equal',
            'f': 2,
            'm': 0.3,
            'rho': 0.0005,
            'b': 1,
            'mu': 0.001,
            'feddyn_alpha': 0.001
        },
        'model_name': 'LeNet5',
        'base_datasize': 20,
        'data_growth_rate': 20
    },
    'client': {
        'local_epoch': 2,
        'batch_size': 50,
        'optimizer':{
            'type': 'SGD',
            'lr': 0.01
            
        }
    },
    'trial_name': 'test2',
    'resume': True

}

init(config=config)

run()
