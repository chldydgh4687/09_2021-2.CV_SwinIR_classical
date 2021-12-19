import argparse
import os

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        #### SwinSR "Classical" Task
        self.parser.add_argument("--name", type=str, default="classicalx2")
        self.parser.add_argument("--scale", type=int, default=2)

        #### TRAIN DATAROOT
        self.parser.add_argument("--tr_H", type=str, default="data/train/trainH")
        self.parser.add_argument("--tr_L", type=str, default="data/train/trainL")
        self.parser.add_argument("--tr_testH", type=str, default="data/test/Set5HR")
        self.parser.add_argument("--tr_testL", type=str, default="data/test/Set5LRX2")

        #### Data Parameters
        self.parser.add_argument("--tr_Hsize", type=int, default=96) # SPLIT PATCH SIZE  ( HR : 96 , LR : 48 )
        self.parser.add_argument("--tr_Shuffle", type=bool, default=True)
        self.parser.add_argument("--tr_num_workers", type=int, default=16)
        self.parser.add_argument("--tr_batch", type=int, default=16) # batch_size / gpu_id in train.py

        #### TRAIN_PARAMETERS
        self.parser.add_argument("--tr_epohcs", type=int, default=1000)
        self.parser.add_argument("--troptim_lr", type=float, default=2e-4)
        self.parser.add_argument("--troptim_lrwd", type=float, default=0) # learning rate weight decay
        self.parser.add_argument("--troptim_clipgrad", type=bool, default=False) #
        self.parser.add_argument("--troptim_reuse",type=bool, default=True ) #

        #self.parser.add_arugmnet("")
        #e_decay = 0.999

        self.parser.add_argument("--lr_scheduler", type=str, default="MultiStepLR")
        self.parser.add_argument("--lr_milestones1", type=int, default=250000)
        self.parser.add_argument("--lr_milestones2", type=int, default=400000)
        self.parser.add_argument("--lr_milestones3", type=int, default=450000)
        self.parser.add_argument("--lr_milestones4", type=int, default=475000)
        self.parser.add_argument("--lr_milestones5", type=int, default=500000)
        self.parser.add_argument("--lr_gamma",type=float, default=0.5)

        # regularizer_orthstep : null
        # regularizer_clipstep : null

        # param_strict : true
        # param_strict : true

        #### Checkpoints Parameters
        self.parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
        self.parser.add_argument("--checkpoint_test", type=int, default=5000)
        self.parser.add_argument("--checkpoint_save", type=int, default=5000)
        self.parser.add_argument("--checkpoint_print", type=int, default=200)


        #### Parameters in Original Code


    def print_option(self,options,save_path):
        args = vars(options)

        common_list = {}
        training_list = {}
        test_list = {}
        for k, v in sorted(args.items()):
            common_list[k] = v

        parameter_log = save_path + "/" + options.name + "_opt.txt"
        with open(parameter_log, "wt") as opt_file:
            for k, v in sorted(common_list.items()):
                opt_file.write('%s : %s\n' % (str(k), str(v)))

        opt_file.close()

    def parse(self):

        self.opt = self.parser.parse_args()

        checkpoints_dir = self.opt.checkpoint_dir  + "/" + self.opt.name
        try:
            os.mkdir(checkpoints_dir)
        except:
            pass

        print("parsing training options!")
        self.print_option(self.opt, checkpoints_dir)

        return self.opt
