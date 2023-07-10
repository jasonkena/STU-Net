import os
import pickle
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict



def generate_splits_ct(output_path):
    base_path = "/data/adhinart/bonnet/dataset/UniHeart/MMWHS-CT"
    csv = pd.read_csv(os.path.join(base_path, "info.csv"))
    # splits should be of form mmwhsct_001

    train = []
    val = []
    for i, uid in enumerate(csv.uid):
        uid = int(uid.split("-")[-1])
        if csv.split[i] == "train":
            train.append(f"mmwhsct_{uid:03}")
        elif csv.split[i] == "test":
            val.append(f"mmwhsct_{uid:03}")
        else:
            raise ValueError("Unknown split")
    
    result = OrderedDict()
    result["train"] = np.array(train)
    result["val"] = np.array(val)
    pickle.dump([result], open(output_path, 'wb'))

def generate_splits_mri(output_path):
    base_path = "/data/adhinart/bonnet/dataset/UniHeart/MMWHS-MRI"
    csv = pd.read_csv(os.path.join(base_path, "info.csv"))
    # splits should be of form mmwhsmri_001

    train = []
    val = []
    for i, uid in enumerate(csv.uid):
        uid = int(uid.split("-")[-1])
        if csv.split[i] == "train":
            train.append(f"mmwhsmri_{uid:03}")
        elif csv.split[i] == "test":
            val.append(f"mmwhsmri_{uid:03}")
        else:
            raise ValueError("Unknown split")
    
    result = OrderedDict()
    result["train"] = np.array(train)
    result["val"] = np.array(val)
    pickle.dump([result], open(output_path, 'wb'))

# [OrderedDict([('train', array(['mmwhsct_001', 'mmwhsct_002', 'mmwhsct_003', 'mmwhsct_005',
#        'mmwhsct_006', 'mmwhsct_007', 'mmwhsct_008', 'mmwhsct_010',
#        'mmwhsct_011', 'mmwhsct_012', 'mmwhsct_013', 'mmwhsct_015',
#        'mmwhsct_016', 'mmwhsct_018', 'mmwhsct_019', 'mmwhsct_020'],
#       dtype='<U11')), ('val', array(['mmwhsct_004', 'mmwhsct_009', 'mmwhsct_014', 'mmwhsct_017'],
#       dtype='<U11'))]), OrderedDict([('train', array(['mmwhsct_002', 'mmwhsct_003', 'mmwhsct_004', 'mmwhsct_005',
#        'mmwhsct_006', 'mmwhsct_007', 'mmwhsct_008', 'mmwhsct_009',
#        'mmwhsct_010', 'mmwhsct_012', 'mmwhsct_014', 'mmwhsct_015',
#        'mmwhsct_017', 'mmwhsct_018', 'mmwhsct_019', 'mmwhsct_020'],
#       dtype='<U11')), ('val', array(['mmwhsct_001', 'mmwhsct_011', 'mmwhsct_013', 'mmwhsct_016'],
#       dtype='<U11'))]), OrderedDict([('train', array(['mmwhsct_001', 'mmwhsct_002', 'mmwhsct_003', 'mmwhsct_004',
#        'mmwhsct_005', 'mmwhsct_006', 'mmwhsct_009', 'mmwhsct_010',
#        'mmwhsct_011', 'mmwhsct_013', 'mmwhsct_014', 'mmwhsct_015',
#        'mmwhsct_016', 'mmwhsct_017', 'mmwhsct_019', 'mmwhsct_020'],
#       dtype='<U11')), ('val', array(['mmwhsct_007', 'mmwhsct_008', 'mmwhsct_012', 'mmwhsct_018'],
#       dtype='<U11'))]), OrderedDict([('train', array(['mmwhsct_001', 'mmwhsct_002', 'mmwhsct_003', 'mmwhsct_004',
#        'mmwhsct_005', 'mmwhsct_006', 'mmwhsct_007', 'mmwhsct_008',
#        'mmwhsct_009', 'mmwhsct_011', 'mmwhsct_012', 'mmwhsct_013',
#        'mmwhsct_014', 'mmwhsct_016', 'mmwhsct_017', 'mmwhsct_018'],
#       dtype='<U11')), ('val', array(['mmwhsct_010', 'mmwhsct_015', 'mmwhsct_019', 'mmwhsct_020'],
#       dtype='<U11'))]), OrderedDict([('train', array(['mmwhsct_001', 'mmwhsct_004', 'mmwhsct_007', 'mmwhsct_008',
#        'mmwhsct_009', 'mmwhsct_010', 'mmwhsct_011', 'mmwhsct_012',
#        'mmwhsct_013', 'mmwhsct_014', 'mmwhsct_015', 'mmwhsct_016',
#        'mmwhsct_017', 'mmwhsct_018', 'mmwhsct_019', 'mmwhsct_020'],
#       dtype='<U11')), ('val', array(['mmwhsct_002', 'mmwhsct_003', 'mmwhsct_005', 'mmwhsct_006'],
#       dtype='<U11'))])]
        
TASKS = {"mmwhs_ct": generate_splits_ct, "mmwhs_mri": generate_splits_mri}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", help="Path to save the splits JSON file")
    parser.add_argument("--task", type=str, help="Task to perform")
    args = parser.parse_args()
    
    TASKS[args.task](args.output_path)
