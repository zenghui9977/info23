# -*- coding: utf-8 -*-
from collections import defaultdict
import csv
import argparse

def args_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--exp_name', type=str, help='the exp_name expected to set one of the list: [our, fixed, x_based, loss_based, loss_and_x_based]')
    parser.add_argument('--MC', type=str, help='MC name, [MC1, MC2, MC3]')

    args = parser.parse_args()
    return args


def read_data_from_csv(file_path):
    
    result = defaultdict(list)
    with open(file_path, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            
            result[row[0]].extend(list(map(float,row[1:])))
    # print(result)
    return result


# read_data_from_csv('./params/new_new_new_coco.csv')