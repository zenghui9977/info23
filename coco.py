import os
import random
import time
import gc
import numpy as np
import torch
import csv
from torch.utils.data import Subset, RandomSampler, SequentialSampler, BatchSampler, DataLoader

from fllib.base import logger, BaseFL, init_config, init_logger, set_all_random_seed
from fllib.server.base import BaseServer
from fllib.client.base import BaseClient
from fllib.server.visualization import vis_scalar
from fllib.datasets.base import FederatedDataset, max_limit_list, robust_cycle_list
from fllib.datasets.simulation import size_of_division
from fllib.datasets.utils import collate_fn

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from collections import defaultdict

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from prettytable import PrettyTable



GLOBAL_ROUND = 'Round'
GLOBAL_ACC = 'Accuracy'
GLOBAL_LOSS = 'Loss'
GLOBAL_TIME = 'Time'


class OurClient(BaseClient):
    def __init__(self, config, device):
        super(OurClient, self).__init__(config, device)

    def train(self, client_id, local_trainset):

        # start_time = time.time()
        _, optimizer = self.train_preparation()

        self.local_model.train()

        for e in range(self.config.local_epoch):
            batch_loss = []

            for imgs, labels in local_trainset:

                imgs = list(img.to(self.device) for img in imgs)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

                
                loss_dict = self.local_model(imgs, labels)

                losses = sum(loss for loss in loss_dict.values())
                # print(losses)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                batch_loss.append(losses.item())
            
            current_epoch_loss = np.mean(batch_loss)
            logger.debug('Client: {}, local epoch: {}, loss: {:.4f}'.format(client_id, e, current_epoch_loss))
        # train_time = time.time() - start_time
        # logger.debug('Client: {}, training {:.4f}s'.format(client_id, train_time))



class OurServer(BaseServer):
    def __init__(self, config, clients, client_class, global_model, fl_trainset, testset, device, current_round=0, records_save_filename=None, vis=None):
        super(OurServer, self).__init__(config, clients, client_class, global_model, fl_trainset, testset, device, current_round, records_save_filename, vis)

        self.accumulate_time = 0
        self.computation_capacity = config.computation_capacity
        self.bandwidth = [i * config.bandwidth_magnitude for i in config.bandwidth]
        
        self.training_time_limitation = config.training_time_limitation
        self.base_datasize = config.base_datasize
        self.data_growth_rate = config.data_growth_rate

        self.process_f_b()

        self.model_size = sum([param.nelement() for param in global_model.parameters()])

    def multiple_steps(self):
        for _ in range(self.config.rounds):
            self.one_step()
            self.save_the_checkpoint(save_path=self.config.records_save_folder, save_file_name=self.records_save_filename + '_checkpoint')
            
            gc.collect()
            torch.cuda.empty_cache()

            if self.accumulate_time >= self.training_time_limitation:
                logger.debug('Stop training......')
                return self.global_model

        return self.global_model

    def one_step(self):
        logger.info('----- Round {}th -----'.format(self.current_round))

        self.client_selection(clients=self.clients, clients_per_round=self.config.clients_per_round)
        _, clients_training_time = self.client_training()
        

        self.accumulate_time += max(clients_training_time.values())

        self.aggregation()
        self.update_global_model()
        test_result = self.test()


        # self.train_records[GLOBAL_ROUND] = self.current_round 
        # self.train_records[GLOBAL_ACC] = acc
        # self.train_records[GLOBAL_LOSS] = loss

        # self.write_one_row(one_raw=[self.current_round, acc, loss], save_path=self.config.records_save_folder, save_file_name=self.records_save_filename)
        
        self.train_records = test_result
        self.transfer_train_records()

        self.show_dict_as_table()


        
        self.train_records[GLOBAL_ROUND] = self.current_round
        self.train_records[GLOBAL_TIME] = self.accumulate_time

        self.is_cvs_header_correct(list(self.train_records.keys()), save_path=self.config.records_save_folder, save_file_name=self.records_save_filename)

        self.write_one_dictrow(one_raw=self.train_records, save_path=self.config.records_save_folder, save_file_name=self.records_save_filename)


        logger.debug('{}th round use {:.4f}s.'.format(self.current_round, max(clients_training_time.values())))
        if self.vis is not None:
            vis_scalar(vis=self.vis, figure_name=f'{self.config.records_save_folder}/{self.records_save_filename}/{GLOBAL_ACC}', scalar_name=GLOBAL_ACC, x=self.current_round, y=acc)
            vis_scalar(vis=self.vis, figure_name=f'{self.config.records_save_folder}/{self.records_save_filename}/{GLOBAL_LOSS}', scalar_name=GLOBAL_LOSS, x=self.current_round, y=loss)
            
        self.current_round += 1

        logger.debug('Accumalated training time {:.4f}s.'.format(self.accumulate_time))
        
        return self.global_model

   
    def client_training(self):
        if len(self.selected_clients) > 0:

            clients_trainig_time = {}

            for client in self.selected_clients:
                start_time = time.time()
                
                # self.train_batchsize = self.computation_capacity[client]

                self.local_updates[client] = {
                    'model': self.client_class.step(
                        global_model = self.global_model,
                        client_id=client,
                        local_trainset=self.fl_trainset.get_dynamic_dataloader(
                            client_id = client,
                            batch_size = self.train_batchsize,
                            current_round = self.current_round,
                            base_datasize = self.base_datasize[client],
                            data_growth_rate = self.data_growth_rate[client]
                        )
                    ).state_dict(),

                    'size': self.fl_trainset.get_dynamic_datasize(
                        client_id = client, 
                        current_round = self.current_round, 
                        base_datasize = self.base_datasize[client], 
                        data_growth_rate = self.data_growth_rate[client]
                    )
                }
                training_time = time.time() - start_time

                training_time = self.train_batchsize * training_time / self.computation_capacity[client]

                trans_time = 2 * self.model_size / self.bandwidth[client]
                
                clients_trainig_time[client] = training_time + trans_time

                logger.info('Client {}, Training time {:.4f}s, Comminication time {:.4f}, Total {:.4f}'.format(client, training_time, trans_time, clients_trainig_time[client]))

        else:
            logger.warning('No clients in this round')
            self.local_updates = None
        return self.local_updates, clients_trainig_time

    def test(self):
        self.global_model.eval()
        self.global_model.to(self.device)

        mAP_metrics = MeanAveragePrecision(compute_on_cpu=True)

        logger.debug('Test in the server')
        with torch.no_grad():
            for imgs, labels in self.testset:
                imgs = list(img.to(self.device) for img in imgs)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

                preds = self.global_model(imgs)


                _ = mAP_metrics.update(preds, labels)

            test_result_dict = mAP_metrics.compute()
            
            return test_result_dict

            
    def write_header(self, header, save_path, save_file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, save_file_name) + '.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)       

    def is_cvs_header_correct(self, target_header, save_path, save_file_name):
        if not os.path.exists(os.path.join(save_path, save_file_name) + '.csv'):
            self.write_header(target_header, save_path, save_file_name)       
        else:
            with open(os.path.join(save_path, save_file_name) + '.csv', 'r', encoding='utf-8') as f:
                f_csv = csv.reader(f)
                header = next(f_csv)
                               
                if list(set(header).difference(set(target_header))) != []:
                    self.write_header(target_header, save_path, save_file_name)

    def write_one_dictrow(self, one_raw, save_path, save_file_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, save_file_name) + '.csv', 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(one_raw.keys()))
            w.writerow(one_raw)


    def transfer_train_records(self):
        for k, v in self.train_records.items():      
            if torch.is_tensor(v):
                self.train_records[k] = v.item()


    def show_dict_as_table(self, trimmed_num=5):
        # sort first
        sepkeys = sorted(self.train_records.keys())

        frag_keys = [sepkeys[:len(sepkeys)//2], sepkeys[len(sepkeys)//2 :]]
        trimmed_dict = [{k: self.train_records[k] for k in i} for i in frag_keys]

        output_str = 'The results are: '

        for i in trimmed_dict:
            output_str += '\n'
            dicttable = PrettyTable(list(i.keys()))
            dicttable.add_row(list(i.values()))
            dicttable.float_format = '.4'
            # dicttable.set_style(MSWORD_FRIENDLY) 
            output_str += dicttable.get_string()
        

        logger.debug(output_str)
        

    def process_f_b(self):
        self.computation_capacity = {key: value for key, value in zip(self.clients, self.computation_capacity)}
        self.bandwidth = {key: value for key, value in zip(self.clients, self.bandwidth)}
        self.data_growth_rate = {key: value for key, value in zip(self.clients, self.data_growth_rate)}
        self.base_datasize = {key: value for key, value in zip(self.clients, self.base_datasize)}





class OurFL(BaseFL):
    def __init__(self):
        super(OurFL, self).__init__()

    def run(self):
        
        self.global_model = self.server.multiple_steps()


class CocoFL_dataset(FederatedDataset):
    def __init__(self, data_name, trainset, testset, simulated, simulated_root, distribution_type, clients_id, class_per_client=2, alpha=0.9, min_size=1):
        self.trainset = trainset
        self.testset = testset

        self.idx_dict = {}

        self.data_name = data_name
        self.simulated = simulated

        self.simulated_root = simulated_root

        self.distribution_type = distribution_type

        self.clients_id = clients_id
        self.clients_num = len(clients_id)

        if self.distribution_type == 'iid':
            distribution_args = 0
        elif self.distribution_type == 'non_iid_class':
            distribution_args = class_per_client
        elif self.distribution_type == 'non_iid_dir':
            distribution_args = alpha        

        self.store_file_name = f'{self.data_name}_{self.distribution_type}_clients{self.clients_num}_args{distribution_args}'

        if os.path.exists(os.path.join(self.simulated_root, self.store_file_name)) and (not self.simulated):
            logger.info(f'Clients data file {self.store_file_name} already exist. Loading......')
            self.clients_data = torch.load(os.path.join(simulated_root, self.store_file_name))
            
        else:
            if not os.path.exists(self.simulated_root):
                os.makedirs(self.simulated_root)
            logger.info(f'Initialize the file {self.store_file_name}.')
            self.clients_data = self.coco_data_distribution_simulation()
            
            
            torch.save(self.clients_data, os.path.join(self.simulated_root, self.store_file_name))


    def coco_data_distribution_simulation(self):
        clients_data = defaultdict(list)
        data_num = len(self.trainset)

        datasize_per_client = size_of_division(self.clients_num, data_num)
        


        temp = set(range(data_num))

        for c in range(self.clients_num):
            cur_client_id = self.clients_id[c]
            
            rand_set = np.random.choice(list(temp), datasize_per_client[c], replace=False)

            temp = temp - set(rand_set)

            clients_data[cur_client_id].extend(rand_set)

            random.shuffle(clients_data[cur_client_id])

        return clients_data

    def get_dynamic_dataloader(self, client_id, batch_size, current_round, base_datasize, data_growth_rate, istrain=True, drop_last=False):
        if istrain:

            if client_id in self.clients_id:
                data_idx = self.clients_data[client_id]

                dynamic_data_size = base_datasize + current_round * data_growth_rate

                # data_idx = robust_cycle_list(data_idx, dynamic_data_size)
                data_idx = max_limit_list(data_idx, dynamic_data_size)

                sub_dataset = Subset(self.trainset, data_idx)
                

                sampler = RandomSampler(sub_dataset)
                batch_sampler = BatchSampler(sampler, batch_size=min(len(data_idx), batch_size), drop_last=drop_last)

                return DataLoader(sub_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
            else:
                raise ValueError('The client id is not existed.') 


    def get_coco_testset(self, batch_size):
        test_sampler = SequentialSampler(self.testset)

        return DataLoader(self.testset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn)






global_fl = OurFL()

def init(config=None, custome_client_class=None, custome_server=None, global_model=None, custome_fl_dataset=None):

    global global_fl
    
    config = init_config(config)

    init_logger(config.log_level)

    set_all_random_seed(config.seed)

    global_fl.init_fl(config, custome_server=custome_server, custome_client_class=custome_client_class, global_model=global_model, fl_dataset=custome_fl_dataset)




def run():
    global global_fl

    global_fl.run()

config = {
    'dataset': {
        'data_name': 'mini_coco2017',
        'download': False,
        'distribution_type': 'iid',
        'alpha': 0.5,
        'simulated': True,
        
    },
    'server': {
        'clients_num': 5,
        'rounds': 100,
        'clients_per_round': 5,
        'aggregation_rule': 'fedavg',
        'aggregation_detail': {
            'type': 'equal'
        },
        'model_name': 'fastrcnn',

    },
    'client': {
        'local_epoch': 1,
        'test_batch_size': 6,
        'batch_size': 2,
        'optimizer':{
            'type': 'SGD',
            'lr': 0.001      
        }
    },
    'trial_name': 'ours',
    'resume': False,
    'is_visualization': False,

    'computation_capacity': [2, 2, 2, 2, 2],  # f 
    'bandwidth': [1, 1.2, 1.5, 2.0, 4.0], # b
    'bandwidth_magnitude': 1,  # 1e6
    'base_datasize': 100,
    'data_growth_rate': [100, 150, 200, 250, 120],
    'training_time_limitation': 60
}

use_pretrain = False
model = fasterrcnn_resnet50_fpn(pretrained=use_pretrain, progress=False)


from args import args_parser, read_data_from_csv


csv_params = read_data_from_csv('./params/new_new_new_coco.csv')

args = args_parser()

config['trial_name'] = args.exp_name + '_' + args.MC

config['computation_capacity'] = csv_params[args.exp_name + '_f_' + args.MC]
config['bandwidth'] = csv_params[args.exp_name + '_B_' + args.MC]
config['base_datasize'] = list(map(int, csv_params['base_datasize_' + args.MC]))
config['data_growth_rate'] = list(map(int, csv_params['data_growth_rate_' + args.MC]))



init(config=config, custome_server=OurServer, custome_client_class=OurClient, global_model=model, custome_fl_dataset=CocoFL_dataset)

run()


# trial_name_list = ['our', 'fixed', 'x_based', 'loss_based', 'loss_x_based']

# trial_name_index = 0
# mc = 1



# config['trial_name'] = trial_name_list[trial_name_index] + f'MC{mc}'

# if trial_name_index == 0:
#     config['computation_capacity'] = [250, 464.8, 390.6, 328.1, 339.84]
#     config['bandwidth'] = [75, 108.75, 121.25, 96.25, 68.75]
#     config['base_datasize'] = 100
#     config['data_growth_rate'] = [175, 222, 140, 164, 175]

# elif trial_name_index == 1:
#     config['computation_capacity'] = [201.7574, 172.1232, 456.707, 283.051, 330.6735]
#     config['bandwidth'] = [0.74915,	0.618317, 0.780165, 0.975501, 0.904441]
#     config['base_datasize'] = 100
#     config['data_growth_rate'] = [175, 222, 140, 163, 175]

# elif trial_name_index == 2:
#     config['computation_capacity'] = [225.6944, 285.8796, 180.5555, 210.6481, 225.6944]
#     config['bandwidth'] = [0.78, 0.9880, 0.6240, 0.7280, 0.78]
#     config['base_datasize'] = 100
#     config['data_growth_rate'] = [175, 222, 140, 163, 175]
   
# elif trial_name_index == 3:
#     config['computation_capacity'] = [189.8148, 192.5925, 195.3703,	195.3703, 203.7037]
#     config['bandwidth'] = [0.6150, 0.624, 0.633, 0.633, 0.66]
#     config['base_datasize'] = 100
#     config['data_growth_rate'] = [175, 222, 140, 163, 175]
   
# elif trial_name_index == 4:
#     config['computation_capacity'] = [207.7546, 239.2361, 187.963, 203.01, 214.6991]
#     config['bandwidth'] = [0.6975, 0.806, 0.6285, 0.6805, 0.72]
#     config['base_datasize'] = 100
#     config['data_growth_rate'] = [175, 222, 140, 163, 175]




