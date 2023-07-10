from collections import OrderedDict
import SimpleITK as sitk
import shutil
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
import numpy as np
import nibabel as nib

import os
import glob
import argparse

def processed_mmwhs_ct():
    base_path = "/data/adhinart/bonnet/dataset/UniHeart/MMWHS-CT"
    csv = pd.read_csv(join(base_path, "info.csv"))

    mapping = -np.ones(max(MAPPING.keys()) + 1, dtype=int)
    for k, v in MAPPING.items():
        mapping[k] = v

    assert len(set(csv.labels.tolist())) == 1
    # 0_1_2_3_4_5_6_701
    label_string = csv.labels.tolist()[0]
    print(label_string)
    used_labels = np.unique(list(map(int, label_string.split("_"))))
    used_labels = [MAPPING[l] for l in used_labels]
    # zero could be any of the SINGLE labels, except anything referenced
    zero_could_be = list(range(11))
    for l in used_labels:
        # remove all individual labels which are part of hierarchy
        if "/" in LABELS[str(l)]:
            for ll in LABELS[str(l)].split("/"):
                zero_could_be.remove(int(ll))
        else:
            zero_could_be.remove(l)
    zero_could_be.append(0)
    zero_could_be = sorted(zero_could_be)

    # assert set(csv.split) == set(["train", "test"])
    dataset = []

    for i, uid in enumerate(csv.uid):
        uid = uid.split("-")[-1]
        assert (len(uid) == 3)
        image_file = os.path.join(base_path, f"MMWHS-CT-{uid}-image.nii.gz")
        label_file = os.path.join(base_path, f"MMWHS-CT-{uid}-label.nii.gz")

        out_image_file = "mmwhsct_" + uid + "_0000.nii.gz"
        # "_0000.nii.gz"
        out_label_file = "mmwhsct_" + uid + ".nii.gz"

        dataset.append((image_file, label_file, out_image_file, out_label_file))

    return dataset, mapping, zero_could_be, used_labels

def processed_mmwhs_mri():
    base_path = "/data/adhinart/bonnet/dataset/UniHeart/MMWHS-MRI"
    csv = pd.read_csv(join(base_path, "info.csv"))

    mapping = -np.ones(max(MAPPING.keys()) + 1, dtype=int)
    for k, v in MAPPING.items():
        mapping[k] = v

    assert len(set(csv.labels.tolist())) == 1
    # 0_1_2_3_4_5_6_701
    label_string = csv.labels.tolist()[0]
    print(label_string)
    used_labels = np.unique(list(map(int, label_string.split("_"))))
    used_labels = [MAPPING[l] for l in used_labels]
    # zero could be any of the SINGLE labels, except anything referenced
    zero_could_be = list(range(11))
    for l in used_labels:
        # remove all individual labels which are part of hierarchy
        if "/" in LABELS[str(l)]:
            for ll in LABELS[str(l)].split("/"):
                zero_could_be.remove(int(ll))
        else:
            zero_could_be.remove(l)
    zero_could_be.append(0)
    zero_could_be = sorted(zero_could_be)

    # assert set(csv.split) == set(["train", "test"])
    dataset = []
    for i, uid in enumerate(csv.uid):
        uid = uid.split("-")[-1]
        assert (len(uid) == 3)
        image_file = os.path.join(base_path, f"MMWHS-MRI-{uid}-image.nii.gz")
        label_file = os.path.join(base_path, f"MMWHS-MRI-{uid}-label.nii.gz")

        out_image_file = "mmwhsmri_" + uid + "_0000.nii.gz"
        # "_0000.nii.gz"
        out_label_file = "mmwhsmri_" + uid + ".nii.gz"

        dataset.append((image_file, label_file, out_image_file, out_label_file))

    return dataset, mapping, zero_could_be, used_labels



def process_file(image_file, label_file, image_out_file, label_out_file, mapping):
    label = nib.load(label_file).get_fdata()
    shape = label.shape

    label = mapping[label.flatten().astype(int)].reshape(shape)
    assert np.all(label >= 0)

    sitk.WriteImage(sitk.GetImageFromArray(label), label_out_file)

    # since there are spacing shenanigans
    image = nib.load(image_file).get_fdata()
    sitk.WriteImage(sitk.GetImageFromArray(image), image_out_file)
    # shutil.copy(image_file, image_out_file)


# 0=BG: background (0)
# 1=MYO-LV: myocardium of the left ventricle (205)
# 101=LV normal myocardium (200)
# 102=LV myocardial edema (1220)
# 103=LV myocardial scars (2221)
# 2=LV: left ventricle blood cavity (500)
# 3=RV: right ventricle blood cavity (600)
# 4=LA: left atrium blood cavity (420)
# 5=RA: right atrium blood cavity (550)
# 6=PA: pulmonary artery (850)
# 7=AO: aorta
# 701=AAO: ascending aorta (820)
# 702=DAO: descending aorta (822)


# NOTE: monkey patch hierarchical processing code
# used for dataset.json
LABELS = {
    "0":"background", # note that background can be anything not used
    "1": "LV normal myocardium",
    "2": "LV myocardial edema",
    "3": "LV myocardial scars",
    "4": "LV",
    "5": "RV",
    "6": "LA",
    "7":"RA",
    "8":"PA",
    "9":"AAO",
    "10":"DAO",
    "11":"1/2/3",
    "12":"9/10"
}

# used to remap labels in volumes
MAPPING = {
    0: 0,
    101: 1,
    102: 2,
    103: 3,
    1: 11,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8,
    701: 9,
    702: 10,
    7: 12
}

TASKS = {
    "mmwhs_ct": (processed_mmwhs_ct, "Task011_heart_ct", "CT"),
    "mmwhs_mri": (processed_mmwhs_mri, "Task012_heart_mri", "MRI")
}

def run(task):
    print("Task:", task)
    process_function, task_name, modality = TASKS[task]

    nnUNet_raw = os.path.join(os.getenv('nnUNet_raw_data_base'), "nnUNet_raw_data")
    
    image_train_folder = join(nnUNet_raw, task_name, "imagesTr")
    label_train_folder = join(nnUNet_raw, task_name, "labelsTr")

    # use test as validation
    image_test_folder = image_train_folder
    label_test_folder = label_train_folder

    maybe_mkdir_p(image_train_folder)
    maybe_mkdir_p(label_train_folder)

    dataset, mapping, zero_could_be, used_labels = process_function()

    tasks = []
    for image_file, label_file, image_out_file, label_out_file in dataset:
        tasks.append((image_file, label_file, os.path.join(image_train_folder, image_out_file), os.path.join(label_train_folder, label_out_file), mapping))

    with Pool(8) as p:
        res = p.starmap_async(process_file, tasks)
        # res_val = p.starmap_async(copy_files, [(id, input_folder, gt_folder, image_val_folder, label_val_folder) for id in splits["val"]])
        _ = res.get() 
    
    json_dict = OrderedDict()
    json_dict['name'] = "Heart"
    json_dict['description'] = "Heart segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": modality
    }
    json_dict['labels'] = LABELS
    json_dict['numTraining'] = len(dataset)
    json_dict['training'] = [{'image': tasks[i][2].replace("_0000",""), "label": tasks[i][3]} for i in range(len(dataset))]
    json_dict['numTest'] = 0
    json_dict['test'] = []
    json_dict['zero_could_be'] = zero_could_be
    json_dict['used_labels'] = used_labels

    save_json(json_dict, os.path.join(nnUNet_raw, task_name, "dataset.json"))
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to perform")
    args = parser.parse_args()

    run(args.task)
