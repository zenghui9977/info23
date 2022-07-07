import os
import time
import gc
import torch
import numpy as np
import torchmetrics
from prettytable import PrettyTable

from fllib.client.base import BaseClient
from fllib.server.base import BaseServer

from fllib.base import logger, BaseFL, init_config, init_logger, set_all_random_seed


GLOBAL_ROUND = 'Round'
GLOBAL_ACC = 'Accuracy'
GLOBAL_LOSS = 'Loss'
GLOBAL_TIME = 'Time'


class OurMNISTClient(BaseClient):
    def __init__(self, config, device):
        super(OurMNISTClient, self).__init__(config, device)

    def train(self, client_id, local_trainset):

        loss_fn, optimizer = self.train_preparation()

        train_accuracy = torchmetrics.Accuracy().to(self.device)

        for e in range(self.config.local_epoch):
            batch_loss = []

            train_accuracy.reset()

            for imgs, labels in local_trainset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.local_model(imgs)

                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                _ = train_accuracy(outputs, labels)

            current_epoch_loss = np.mean(batch_loss)
            current_epochs_acc = train_accuracy.compute().item()
            
            logger.debug('Client: {}, local epoch: {}, loss: {:.4f}, accuracy: {:.4f}'.format(client_id, e, current_epoch_loss, current_epochs_acc))


class OurMNISTServer(BaseServer):
    def __init__(self, config, clients, client_class, global_model, fl_trainset, testset, device, current_round=0, records_save_filename=None, vis=None):
        super(OurMNISTServer, self).__init__(config, clients, client_class, global_model, fl_trainset, testset, device, current_round, records_save_filename, vis)

        self.accumulate_time = 0
        self.computation_capacity = config.computation_capacity
        self.bandwidth = [i * config.bandwidth_magnitude for i in config.bandwidth]
        
        self.training_time_limitation = config.training_time_limitation
        self.base_datasize = config.base_datasize
        self.data_growth_rate = config.data_growth_rate

        self.process_f_b()

        self.model_size = sum([param.nelement() for param in global_model.parameters()])

    def process_f_b(self):
        self.computation_capacity = {key: value for key, value in zip(self.clients, self.computation_capacity)}
        self.bandwidth = {key: value for key, value in zip(self.clients, self.bandwidth)}
        self.data_growth_rate = {key: value for key, value in zip(self.clients, self.data_growth_rate)}


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

        self.train_records = test_result

        self.show_dict_as_table()
        self.train_records[GLOBAL_ROUND] = self.current_round
        self.train_records[GLOBAL_TIME] = self.accumulate_time


        self.is_cvs_header_correct(list(self.train_records.keys()), save_path=self.config.records_save_folder, save_file_name=self.records_save_filename)

        self.write_one_dictrow(one_raw=self.train_records, save_path=self.config.records_save_folder, save_file_name=self.records_save_filename)
    
    
        logger.debug('{}th round use {:.4f}s.'.format(self.current_round, max(clients_training_time.values())))

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
                            base_datasize = self.base_datasize,
                            data_growth_rate = self.data_growth_rate[client]
                        )
                    ).state_dict(),

                    'size': self.fl_trainset.get_dynamic_datasize(
                        client_id = client, 
                        current_round = self.current_round, 
                        base_datasize = self.base_datasize, 
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

        logger.debug('Test in the server')

        loss_fn = self.load_loss_function()
        test_accuracy = torchmetrics.Accuracy().to(self.device)

        test_result_dict = {}

        with torch.no_grad():
            batch_loss = []
            for imgs, labels in self.testset:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                outputs = self.global_model(imgs)
                batch_loss.append(loss_fn(outputs, labels).item())

                _ = test_accuracy(outputs, labels)
            
            total_test_loss = np.mean(batch_loss)
            total_test_accuracy = test_accuracy.compute().item()

            test_result_dict[GLOBAL_LOSS] = total_test_loss
            test_result_dict[GLOBAL_ACC] = total_test_accuracy

            # logger.info('Loss: {:.4f}, Accuracy: {:.4f}'.format(total_test_loss, total_test_accuracy))

            return test_result_dict

    def show_dict_as_table(self, trimmed_num=5):


        output_str = 'The results are: '
        if len(self.train_records.keys()) <= trimmed_num:
            
            dicttable = PrettyTable(list(self.train_records.keys()))
            dicttable.add_row(list(self.train_records.values()))
            dicttable.float_format = '.4'
            output_str += '\n' + dicttable.get_string()
            
        else:
            # sort first
            sepkeys = sorted(self.train_records.keys())

            frag_keys = [sepkeys[:len(sepkeys)//2], sepkeys[len(sepkeys)//2 :]]
            trimmed_dict = [{k: self.train_records[k] for k in i} for i in frag_keys]
            

            for i in trimmed_dict:
                output_str += '\n'
                dicttable = PrettyTable(list(i.keys()))
                dicttable.add_row(list(i.values()))
                dicttable.float_format = '.4'
                # dicttable.set_style(MSWORD_FRIENDLY) 
                output_str += dicttable.get_string()
            

        logger.debug(output_str)

        
class OurMNISTFL(BaseFL):
    def __init__(self):
        super(OurMNISTFL, self).__init__()

    def run(self):
        
        self.global_model = self.server.multiple_steps()




global_fl = OurMNISTFL()

def init(config=None, custome_client_class=None, custome_server=None, global_model=None, custome_fl_dataset=None):

    global global_fl
    
    config = init_config(config)

    init_logger(config.log_level)

    set_all_random_seed(config.seed)

    global_fl.init_fl(config, custome_server=custome_server, custome_client_class=custome_client_class, global_model=global_model, fl_dataset=custome_fl_dataset)


def run():
    global global_fl

    global_fl.run()





#  Our        
config = {
    'dataset': {
        'data_name': 'mnist',
        'download': False,
        'distribution_type': 'iid',
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
        'model_name': 'LeNet5',

    },
    'client': {
        'local_epoch': 2,
        'test_batch_size': 6,
        'batch_size': 2,
        'optimizer':{
            'type': 'SGD',
            'lr': 0.001      
        }
    },
    'trial_name': 'our', #
    'resume': False,
    'is_visualization': False,

    'computation_capacity': [412.415, 501.7007, 471.9388, 446.4286, 578.2313],  # f 
    'bandwidth': [1.0875, 1.1125, 1.375, 1.4, 1.175], # b
    'bandwidth_magnitude': 100000, #1e5
    'base_datasize': 10,
    'data_growth_rate': [23, 29, 23, 18, 23],
    'training_time_limitation': 30
}


trial_name_list = ['ours', 'fixed', 'x_based', 'loss_based', 'loss_and_x_based']

trial_name_index = 4

config['trial_name'] = trial_name_list[trial_name_index]

if trial_name_index == 0:
    config['computation_capacity'] = [19.33195, 23.51722, 22.12213, 20.92634, 27.10459]
    config['bandwidth'] = [1.0875, 1.1125, 1.375, 1.4, 1.175]
    config['base_datasize'] = 10
    config['data_growth_rate'] = [18, 12, 21, 14, 15]

elif trial_name_index == 1:
    config['computation_capacity'] = [24.30296, 9.676883, 6.737472, 31.07206, 24.43898]
    config['bandwidth'] = [1.146016, 0.80916, 0.87422, 0.66450, 0.86756]
    config['base_datasize'] = 10
    config['data_growth_rate'] = [18, 12, 21, 14, 15]

elif trial_name_index == 2:
    config['computation_capacity'] = [14.9473852, 9.964923469, 17.43861607, 11.95790864, 12.45615434]
    config['bandwidth'] = [0.9750, 0.650, 1.1375, 0.78, 0.8125]
    config['base_datasize'] = 10
    config['data_growth_rate'] = [18, 12, 21, 14, 15]
   
elif trial_name_index == 3:
    config['computation_capacity'] = [12.87468167, 14.22991045, 14.22991045, 19.65082911, 20.3284435]
    config['bandwidth'] = [0.6650, 0.7350, 0.7350, 1.0150, 1.05]
    config['base_datasize'] = 10
    config['data_growth_rate'] = [18, 12, 21, 14, 15]
   
elif trial_name_index == 4:
    config['computation_capacity'] = [12.41629516, 11.10092437, 14.09040213, 14.60857742, 15.14668348]
    config['bandwidth'] = [0.76375, 0.6550, 0.87062, 0.8525, 0.884375]
    config['base_datasize'] = 10
    config['data_growth_rate'] = [18, 12, 21, 14, 15]
   

init(config=config, custome_server=OurMNISTServer, custome_client_class=OurMNISTClient)

run()

