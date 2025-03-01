import torch
from nnunet.utilities.nd_softmax import softmax_helper
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.heart import remap_output, remap_target, remap_loss
from nnunet.network_architecture.STUNet import STUNet
from torchinfo import summary

class STUNetTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 0.01
        self.max_num_epochs = 1000
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False
        
    def process_plans(self, plans):
        super().process_plans(plans)
        self.net_conv_kernel_sizes = [[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]
        if len(self.net_num_pool_op_kernel_sizes)>5:
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:5]
        while len(self.net_num_pool_op_kernel_sizes)<5:
            self.net_num_pool_op_kernel_sizes.append([1,1,1])
        
    def initialize_network(self):
        self.network = STUNet(self.num_input_channels, self.num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                                pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes, conv_kernel_sizes = self.net_conv_kernel_sizes)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None
        summary(self.network, input_size=(1, self.num_input_channels, 128, 128, 128))


class STUNetTrainer_small(STUNetTrainer):
    def initialize_network(self):
        self.network =  STUNet(self.num_input_channels, self.num_classes, depth=[1,1,1,1,1,1], dims=[16, 32, 64, 128, 256, 256],
                                pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes, conv_kernel_sizes = self.net_conv_kernel_sizes)
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper 


class STUNetTrainer_base(STUNetTrainer):
    def initialize_network(self):
        self.network =  STUNet(self.num_input_channels, self.num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                                pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes, conv_kernel_sizes = self.net_conv_kernel_sizes)
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper 
        
class STUNetTrainer_large(STUNetTrainer):
    def initialize_network(self):
        self.network =  STUNet(self.num_input_channels, self.num_classes, depth=[2,2,2,2,2,2], dims=[64, 128, 256, 512, 1024, 1024],
                                pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes, conv_kernel_sizes = self.net_conv_kernel_sizes)
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

class STUNetTrainer_huge(STUNetTrainer):
    def initialize_network(self):
        self.network =  STUNet(self.num_input_channels, self.num_classes, depth=[3,3,3,3,3,3], dims=[96, 192, 384, 768, 1536, 1536],
                                pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes, conv_kernel_sizes = self.net_conv_kernel_sizes)
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        
class STUNetTrainer_small_ft(STUNetTrainer_small):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-3

class HeartTrainer(STUNetTrainer_small_ft):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, labels=None, zero_could_be=None, used_labels=None):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.labels = labels
        self.zero_could_be = zero_could_be
        self.used_labels = used_labels

        self.loss = remap_loss(self.loss, labels, zero_could_be, used_labels)

    def run_online_evaluation(self, output, target):
        # defined in nnUNetTrainerV2
        # output shape: [torch.Size([2, 11, 128, 128, 128]), torch.Size([2, 11, 64, 64, 64]), torch.Size([2, 11, 32, 32, 32]), torch.Size([2, 11, 16, 16, 16]), torch.Size([2, 11, 8, 8, 8])]
        # target shape: [torch.Size([2, 1, 128, 128, 128]), torch.Size([2, 1, 64, 64, 64]), torch.Size([2, 1, 32, 32, 32]), torch.Size([2, 1, 16, 16, 16]), torch.Size([2, 1, 8, 8, 8])]
        # evaluation does not depend on self.loss, it calculates dice loss
        output = [remap_output(o, self.labels, self.zero_could_be, self.used_labels) for o in output]
        target = [remap_target(t, self.used_labels) for t in target]

        return super().run_online_evaluation(output, target)

class STUNetTrainer_base_ft(STUNetTrainer_base):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-3


class STUNetTrainer_large_ft(STUNetTrainer_large):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-3

class STUNetTrainer_huge_ft(STUNetTrainer_huge):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-3
