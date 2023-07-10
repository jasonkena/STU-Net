# postprocess npz files from nnUNet_predict

import os
import glob
import json
import argparse
import pickle

from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import nibabel as nib

import torchmetrics

# nnunet/training/network_training
from nnunet.training.network_training.heart import remap_output, remap_target


def read_dataset_json(dataset_json):
    dataset_json = json.load(open(args.dataset_json))

    labels = dataset_json["labels"]
    zero_could_be = dataset_json["zero_could_be"]
    used_labels = dataset_json["used_labels"]

    return labels, zero_could_be, used_labels, dataset_json


def postprocess(input_folder, dataset_json):
    # used only to save final predictions for 3DSlicer
    # not used to compute metrics

    labels, zero_could_be, used_labels, _ = read_dataset_json(dataset_json)
    used_labels = sorted(used_labels)
    print("used_labels:", used_labels)
    print("labels:", labels)
    print("zero_could_be:", zero_could_be)

    files = sorted(glob.glob(os.path.join(input_folder, "*.npz")))
    files = [f for f in files if "final" not in f]

    for f in tqdm(files):
        # since remap_output expects [B, N, 128, 128, 128]
        softmax = torch.from_numpy(np.load(f)["softmax"]).unsqueeze(0)
        remapped = (
            remap_output(softmax, labels, zero_could_be, used_labels).squeeze(0).numpy()
        )
        # crashes on int for some reason
        remapped = np.argmax(remapped, axis=0).astype(np.short)

        # Save remapped array to a nibabel file
        # https://stackoverflow.com/questions/28330785/creating-a-nifti-file-from-a-numpy-array
        img = nib.Nifti1Image(remapped, np.eye(4))
        nib.save(img, f.replace(".npz", "_final.nii.gz"))


def metrics(input_folder, dataset_json, splits_final):
    # used only to save final predictions for 3DSlicer
    # not used to compute metrics

    # whether to transpose the predictions
    reverse = True

    with open(splits_final, "rb") as f:
        splits_final = pickle.load(f)
    train_ids = splits_final[0]["train"].tolist()
    val_ids = splits_final[0]["val"].tolist()

    labels, zero_could_be, used_labels, dataset_json = read_dataset_json(dataset_json)
    used_labels = sorted(used_labels)
    print("used_labels:", used_labels)
    print("labels:", labels)
    print("zero_could_be:", zero_could_be)

    label_files = [x["label"] for x in dataset_json["training"]]
    pred_files = sorted(glob.glob(os.path.join(input_folder, "*.npz")))
    pred_files = [f for f in pred_files if "final" not in f]

    cross_entropy_losses = []
    dices = []

    for f in tqdm(train_ids + val_ids):
        pred_file = [x for x in pred_files if f in x]
        assert len(pred_file) == 1
        pred_file = pred_file[0]

        label_file = [x for x in label_files if f in x]
        assert len(label_file) == 1
        label_file = label_file[0]

        softmax = torch.from_numpy(np.load(pred_file)["softmax"]).unsqueeze(0)
        remapped = remap_output(softmax, labels, zero_could_be, used_labels).type(
            torch.float32
        )
        if reverse:
            remapped = remapped.permute(0, 1, 4, 3, 2)

        label = nib.load(label_file).get_fdata().astype(int)
        label = remap_target(
            torch.from_numpy(label).contiguous(), used_labels
        ).unsqueeze(0)

        cross_entropy_losses.append(F.cross_entropy(remapped, label).item())
        dices.append(
            torchmetrics.functional.dice(
                remapped,
                label,
                average="macro",
                ignore_index=0,
                num_classes=remapped.shape[1],
            )
        )  # average of per-class dice
    print("average train cross entropy loss:", np.mean(cross_entropy_losses[:len(train_ids)]))
    print("average val cross entropy loss:", np.mean(cross_entropy_losses[len(train_ids):]))
    print("average train dice:", np.mean(dices[:len(train_ids)]))
    print("average val dice:", np.mean(dices[len(train_ids):]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Specify the task")
    parser.add_argument("--dataset_json", type=str, help="used for hotpatching")
    parser.add_argument("--input_folder", help="Specify path to npz files")
    # parser.add_argument("--gt_folder", help="Specify path to ground truth files")
    parser.add_argument("--splits_final", help="Specify path to splits_final.pkl")
    args = parser.parse_args()

    if args.task == "postprocess":
        postprocess(args.input_folder, args.dataset_json)
    elif args.task == "metrics":
        metrics(args.input_folder, args.dataset_json, args.splits_final)
    else:
        raise ValueError("Task not supported")
