import time
import os
import torch
import wandb
import logging
import random

from util import args
from util import utils_logger
from util import utils_dataloader
from util import utils

from model.model_utils import model_utils
## classical_SwinIR

def train():

    opt = args.Options().parse()

    # wandb monitoring
    #wandb.init(project="SwinSR")
    #wandb.run.name = opt.name

    # logger monitoring

    logger_name = "train"
    utils_logger.logger_info(logger_name, os.path.join(opt.checkpoint_dir, opt.name + "_" + logger_name +".log"))
    logger = logging.getLogger(logger_name)

    seed = random.randint(1,10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ####### Define Datasets

    SR_Dataloader = utils_dataloader.SR_Dataloader(opt)
    train_loader, test_loader = SR_Dataloader.define_dataloader()

    ###### Define Model
    model = model_utils(opt)
    model.define_model("SwinIR")
    model.init_train()
    logger.info(model.describe_network())

    # for i, train_data in enumerate(train_loader):
    #     print(train_data)

    current_step = 0
    for epoch in range(opt.tr_epohcs):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            model.update_learning_rate(current_step)
            model.get_data(train_data)
            model.optimize_parameters(current_step)

            # log
            if current_step % opt.checkpoint_print == 0 :
                logs = model.current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += "{:s}: {:.3e}".format(k,v)
                logger.info(message)

            # save
            if current_step % opt.checkpoint_save == 0 :
                logger.info("Saving the model...!")
                model.save(current_step)

            # test
            if current_step % opt.checkpoint_test == 0 :
                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data["L_path"][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join("output_test", img_name)
                    utils.mkdir(img_dir)

                    model.get_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = utils.tensor2uint(visuals["E"])
                    H_img = utils.tensor2uint(visuals["H"])

                    save_img_path = os.path.join(img_dir, "{:s}_{:d}.png".format(img_name,current_step))
                    utils.imsave(E_img, save_img_path)

                    current_psnr = utils.calculate_psnr(E_img, H_img, border=opt.scale)

                    logger.info("{:->4d}--> {:>10s} | {:<4.2f}db".format(idx, image_name_ext, current_psnr))
                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                logger.info("<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}db\n".format(epoch,current_step, avg_psnr))





if __name__ == "__main__":
    train()


