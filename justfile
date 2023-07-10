set export
# use scratch directory because NFS is painfully slow
nnUNet_raw_data_base := "/scratch/adhinart/heart_raw"
nnUNet_preprocessed := "/scratch/adhinart/heart_preprocessed"
RESULTS_FOLDER := "/scratch/adhinart/heart_results"
nnunet_use_progress_bar := "1"

default:
    just --list

dataset_ct:
    python nnUNet-1.7.1/nnunet/dataset_conversion/heart_base.py --task mmwhs_ct

dataset_mri:
    python nnUNet-1.7.1/nnunet/dataset_conversion/heart_base.py --task mmwhs_mri

preprocess_ct:
    nnUNet_plan_and_preprocess -t 11 --verify_dataset_integrity

preprocess_mri:
    nnUNet_plan_and_preprocess -t 12 --verify_dataset_integrity

generate_splits_ct:
    python generate_splits.py --task mmwhs_ct --output_path /scratch/adhinart/heart_preprocessed/Task011_heart_ct/splits_final.pkl

generate_splits_mri:
    python generate_splits.py --task mmwhs_mri --output_path /scratch/adhinart/heart_preprocessed/Task012_heart_mri/splits_final.pkl

train_ct:
    CUDA_VISIBLE_DEVICES=0 python run_finetuning.py 3d_fullres HeartTrainer 11 0 -pretrained_weights small_ep4k.model --dataset_json /scratch/adhinart/heart_raw/nnUNet_raw_data/Task011_heart_ct/dataset.json --continue_training

train_mri:
    # run this twice
    CUDA_VISIBLE_DEVICES=1 python run_finetuning.py 3d_fullres HeartTrainer 12 0 -pretrained_weights small_ep4k.model --dataset_json /scratch/adhinart/heart_raw/nnUNet_raw_data/Task012_heart_mri/dataset.json --continue_training

train_ct_bbox:
    pass

old_inference_ct:
    # old uses "all" fold instead of "0"
    # npz since we need to do hierarchical grouping
    RESULTS_FOLDER=/scratch/adhinart/heart_results_bak CUDA_VISIBLE_DEVICES=2 nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task011_heart_ct/imagesTr -o $RESULTS_FOLDER/old_ct -t 11 -m 3d_fullres -f all -tr HeartTrainer -chk model_latest --save_npz

old_inference_mri:
    RESULTS_FOLDER=/scratch/adhinart/heart_results_bak CUDA_VISIBLE_DEVICES=3 nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task012_heart_mri/imagesTr -o $RESULTS_FOLDER/old_mri -t 12 -m 3d_fullres -f all -tr HeartTrainer -chk model_latest --save_npz

shell_expansion_example:
    # os.getenv will see "test", but $RESULTS_FOLDER is expanded immediately
    RESULTS_FOLDER="test" python - $RESULTS_FOLDER/welp

postprocess_ct: 
    # output hierarchical output
    python postprocess.py --task postprocess --dataset_json /scratch/adhinart/heart_raw/nnUNet_raw_data/Task011_heart_ct/dataset.json --input_folder $RESULTS_FOLDER/old_ct

postprocess_mri:
    python postprocess.py --task postprocess --dataset_json /scratch/adhinart/heart_raw/nnUNet_raw_data/Task012_heart_mri/dataset.json --input_folder $RESULTS_FOLDER/old_mri

metrics_ct:
    python postprocess.py --task metrics --dataset_json /scratch/adhinart/heart_raw/nnUNet_raw_data/Task011_heart_ct/dataset.json --input_folder $RESULTS_FOLDER/old_ct --splits_final /scratch/adhinart/heart_preprocessed/Task011_heart_ct/splits_final.pkl

metrics_mri:
    python postprocess.py --task metrics --dataset_json /scratch/adhinart/heart_raw/nnUNet_raw_data/Task012_heart_mri/dataset.json --input_folder $RESULTS_FOLDER/old_mri --splits_final /scratch/adhinart/heart_preprocessed/Task012_heart_mri/splits_final.pkl
