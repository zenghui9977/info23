
import logging
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from fllib.datasets.simulation import data_distribution_simulation
from fllib.datasets.utils import BuildMiniCOCO, MyCocoTransform, MyCompose, MyToTensor, TransformDataset, ConvertCocoPolysToMask, CocoDetection, coco_remove_images_without_annotations


support_dataset = ['mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10', 'cifar100', 'coco2017', 'mini_coco2017']

logger = logging.getLogger(__name__)


def robust_cycle_list(input_list, output_size):
    input_list_len = len(input_list)

    if input_list_len < output_size: 
        temp_times = output_size//input_list_len

        for _ in range(temp_times):
            input_list = np.concatenate((input_list, input_list))
        
    output_list = input_list[:output_size]
    
    return output_list


def max_limit_list(input_list, output_size):
    input_list_len = len(input_list)

    if input_list_len < output_size:
        return input_list
    else:
        return input_list[:output_size]


class BaseDataset(object):
    '''The internal base dataset class, most of the dataset is based on the torch lib

    Args:
        type: The dataset name, options: mnist, fmnist, kmnist, emnist,cifar10
        root: The root directory of the dataset folder.
        download: The dataset should be download or not

    '''
    def __init__(self, datatype, root, download):
        self.type = datatype
        self.root = root
        self.download = download
        self.trainset = None
        self.testset = None
        self.idx_dict = {}
        
        # self.support_dataset = ['mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10', 'cifar100']
        # self.get_dataset()
        

    def get_dataset(self):
        if self.type == 'mnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            self.trainset = torchvision.datasets.MNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.MNIST(root=self.root, train=False, transform=simple_transform, download=self.download)
        
        elif self.type == 'fmnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),])
            self.trainset = torchvision.datasets.FashionMNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.FashionMNIST(root=self.root, train=False, transform=simple_transform, download=self.download)
        
        elif self.type == 'kmnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
            self.trainset = torchvision.datasets.KMNIST(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.KMNIST(root=self.root, train=False, transform=simple_transform, download=self.download)

        elif self.type == 'emnist':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor()])
            self.trainset = torchvision.datasets.EMNIST(root=self.root, train=True, split='byclass', transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.EMNIST(root=self.root, train=False, split='byclass', transform=simple_transform, download=self.download)

        elif self.type == 'cifar10':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
            self.trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.CIFAR10(root=self.root, train=False, transform=simple_transform, download=self.download)

        elif self.type == 'cifar100':
            simple_transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
            self.trainset = torchvision.datasets.CIFAR100(root=self.root, train=True, transform=simple_transform, download=self.download)
            self.testset = torchvision.datasets.CIFAR100(root=self.root, train=False, transform=simple_transform, download=self.download)
        
        elif self.type == 'coco2017':
            simple_transform = MyCompose([MyCocoTransform(), MyToTensor()])
            data_mode = 'instances'
            
            mode = 'train'
            file_path = os.path.join(self.root, f'{self.type}/{mode}2017/')
            annFile = os.path.join(self.root, f'{self.type}/annotations/{data_mode}_{mode}2017.json')
            
            self.trainset = CocoDetection(root=file_path, annFile=annFile, transform=simple_transform)

            test_transform = MyCompose([MyCocoTransform(), MyToTensor()])
            mode = 'val'
            file_path = os.path.join(self.root, f'{self.type}/{mode}2017/')
            annFile = os.path.join(self.root, f'{self.type}/annotations/{data_mode}_{mode}2017.json')       
            self.testset = CocoDetection(root=file_path, annFile=annFile, transform=test_transform)

        elif self.type == 'mini_coco2017':
            data_mode = 'instances'
            
            self.type = 'coco2017'

            simple_transform = MyCompose([MyCocoTransform(), MyToTensor()])
            mode = 'train'
            file_path = os.path.join(self.root, f'{self.type}/{mode}2017/')
            annFile = os.path.join(self.root, f'{self.type}/annotations/{data_mode}_{mode}2017.json')

            self.trainset = CocoDetection(root=file_path, annFile=annFile, transform=simple_transform)
            
            self.trainset = coco_remove_images_without_annotations(self.trainset)
                        
            test_transform = MyCompose([MyCocoTransform(), MyToTensor()])
            mode = 'val'
            file_path = os.path.join(self.root, f'{self.type}/{mode}2017/')
            annFile = os.path.join(self.root, f'{self.type}/annotations/{data_mode}_{mode}2017.json')   

            
            
            mini_file_path = os.path.join(self.root, f'{self.type}/mini_{mode}2017/')
            mini_annFile = os.path.join(self.root, f'{self.type}/annotations/mini_{data_mode}_{mode}2017.json')
            if not(os.path.exists(mini_file_path) and os.path.exists(mini_annFile)):
                mini_coco_builder = BuildMiniCOCO(annotation_file=annFile, origin_img_dir=file_path)
                mini_coco_builder.build(tarDir=mini_file_path, tarFile=mini_annFile, N=100)

            self.testset = CocoDetection(root=mini_file_path, annFile=mini_annFile, transform=test_transform)

            self.type = 'mini_coco2017'
        
        elif self.type == 'kitti':

            self.trainset = torchvision.datasets.Kitti(root=self.root, train=True, download=self.download)
            self.testset = torchvision.datasets .Kitti(root=self.root, train=False, download=self.download)



        else:
            raise ValueError(f'Dataset name is not correct, the options are listed as follows: {support_dataset}')

        return self.trainset, self.testset

        

class FederatedDataset(object):

    def __init__(self, data_name, trainset, testset, simulated, simulated_root, distribution_type, clients_id, class_per_client=2, alpha=0.9, min_size=1):
        
        self.trainset = trainset
        self.testset = testset


        self.idx_dict = self.build_idx_dict()
    
 
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
            self.clients_data = data_distribution_simulation(self.clients_id, self.idx_dict, self.distribution_type, class_per_client, alpha, min_size)
            torch.save(self.clients_data, os.path.join(self.simulated_root, self.store_file_name))

        

    def build_idx_dict(self):
        self.idx_dict = {}
        for idx, data in enumerate(self.trainset):
            _, label = data
            if label in self.idx_dict:
                self.idx_dict[label].append(idx)
            else:
                self.idx_dict[label] = [idx]
        return self.idx_dict

    def get_dataloader(self, client_id, batch_size, istrain=True, drop_last=False):
        if self.data_name in support_dataset:
            if istrain:
                if client_id in self.clients_id:
             
                    data_idx = self.clients_data[client_id]
                    imgs, labels = [], []

                    for i in data_idx:
                        imgs.append(self.trainset[i][0])
                        labels.append(self.trainset[i][1])
                
                    return DataLoader(dataset=TransformDataset(imgs, labels), batch_size=min(len(data_idx), batch_size), shuffle=True, drop_last=drop_last)

                else:
                    raise ValueError('The client id is not existed.')
            else:
                return DataLoader(dataset=self.testset, batch_size=batch_size, shuffle=True)
        
        else:
            raise ValueError(f'Dataset name is not correct, the options are listed as follows: {support_dataset}')
    
                
    def get_client_datasize(self, client_id=None):
        if client_id in self.clients_id:
            return len(self.clients_data[client_id])
        else:
            raise ValueError('The client id is not existed.')
    
    def get_total_datasize(self):
        return len(self.trainset)

    def get_num_class(self):
        return len(self.idx_dict)

    def get_client_datasize_list(self):
        return [len(self.clients_data[i]) for i in self.clients_id]
                
    def get_dynamic_dataloader(self, client_id, batch_size, current_round, base_datasize, data_growth_rate, istrain=True, drop_last=False):
        # data_type in support dataset
        if self.data_name in support_dataset:
            # if this function is used to load the trainset
            if istrain:
                # if current input(client id) in the clients id list
                if client_id in self.clients_id:
                    # get the data index of the client whose id is 'client_id'
                    data_idx = self.clients_data[client_id]
                    
                    # compute the dynamic data size
                    dynamic_data_size = base_datasize + current_round * data_growth_rate

                    data_idx = robust_cycle_list(data_idx, dynamic_data_size)
                    
                    # laod the data to the dataloader
                    imgs, labels = [], []

                    for i in data_idx:
                        imgs.append(self.trainset[i][0])
                        labels.append(self.trainset[i][1])
                
                    return DataLoader(dataset=TransformDataset(imgs, labels), batch_size=min(len(data_idx), batch_size), shuffle=True, drop_last=drop_last)

                else:
                    raise ValueError('The client id is not existed.')                    
            
        else:
            raise ValueError(f'Dataset name is not correct, the options are listed as follows: {support_dataset}')

    def get_dynamic_datasize(self, client_id, current_round, base_datasize, data_growth_rate):
        if client_id in self.clients_id:
            return base_datasize + current_round * data_growth_rate
        else:
            raise ValueError('The client id is not existed.')