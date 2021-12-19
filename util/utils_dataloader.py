import numpy as np
import random
from util import utils
from torch.utils.data import DataLoader

# son class : random split batch
class Random_Split_Batch():
    def __init__(self, opt, H_path, L_path, type):
        self.opt = opt
        self.type = type

        self.scale = self.opt.scale
        self.patch_size = self.opt.tr_Hsize
        self.L_patch = self.patch_size // self.scale

        self.paths_H = utils.get_image_paths(H_path)
        self.paths_L = utils.get_image_paths(L_path)

        assert self.paths_H, "Error : H path is empty"
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), "L/H mismatch - {}, {}.".format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):
        # paths_H : Himg_path_list
        # pahts_l : Limg_path_list

        H_paths = self.paths_H[index]
        L_paths = self.paths_L[index]

        img_H = utils.imread_uint(H_paths)
        img_H = utils.uint2stand(img_H)

        # crop w, h had odd size
        img_H = utils.modcrop(img_H, self.scale)


        img_L = utils.imread_uint(L_paths)
        img_L = utils.uint2stand(img_L)

        if self.type == "train":
            H,W,C = img_L.shape

            rnd_h = random.randint(0,max(0, H-self.L_patch))
            rnd_w = random.randint(0,max(0, W-self.L_patch))
            img_L = img_L[rnd_h:rnd_h + self.L_patch,rnd_w:rnd_w + self.L_patch]

            rnd_h_H, rnd_w_H = int(rnd_h * self.scale) , int(rnd_w * self.scale)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size,rnd_w_H:rnd_w_H + self.patch_size]

            mode = random.randint(0,7)
            img_L, img_H = utils.augment_img(img_L, mode=mode), utils.augment_img(img_H, mode=mode)

        # WHEN TYPE=="TEST" NOT OPERATE SPLIT BATCH
        img_H, img_L = utils.single2tensor3(img_H), utils.single2tensor3(img_L)

        return {"L": img_L, "H": img_H, "L_path" : L_paths, "H_path" : H_paths }

    def __len__(self):
        return len(self.paths_H)

# parent class : SR_LOADER
class SR_Dataloader():
    def __init__(self, opt):
        self.opt = opt
        self.tr_batch = self.opt.tr_batch
        self.IsShuffle = self.opt.tr_Shuffle
        self.cpu_worker = self.opt.tr_num_workers

    def define_dataloader(self):

        train_set = Random_Split_Batch(self.opt, self.opt.tr_H, self.opt.tr_L, "train")
        test_set = Random_Split_Batch(self.opt, self.opt.tr_testH, self.opt.tr_testL, "test")

        # for i, train_dat in enumerate(train_set):
        #     print(train_dat["L"].shape, train_dat["H"].shape)

        train_loader = DataLoader(train_set,
                                  batch_size = self.tr_batch,
                                  shuffle=self.IsShuffle,
                                  num_workers= self.cpu_worker,
                                  drop_last=True,
                                  pin_memory=True)

        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=False,
                                 pin_memory=True)

        return train_loader, test_loader





