from collections import OrderedDict
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import torch.optim as optim
from torch.optim import lr_scheduler
from model.SwinIR import SwinIR

class model_utils():
    def __init__(self,opt):
        self.opt = opt
        self.save_dir = opt.checkpoint_dir + "/" + opt.name
        self.device = torch.device("cuda" if torch.cuda.is_available() is not None else "cpu")
        self.schedulers = []
        self.schedule_milestone = [self.opt.lr_milestones1,self.opt.lr_milestones2,self.opt.lr_milestones3,self.opt.lr_milestones4,self.opt.lr_milestones5]
        self.model = None
        self.optimizer = None

    def define_model(self, net_name ):
        if net_name == "SwinIR":
            self.model = SwinIR().to(self.device)
            #print(self.model)

    def init_train(self):
        self.model.train()
        self.log_dict = OrderedDict()
        net_optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                net_optim_params.append(v)
            else:
                print("Params [{:s}] will not optimze".format(k))

        self.optimizer = optim.Adam(net_optim_params,lr=self.opt.troptim_lr, weight_decay=0)
        self.schedulers.append(lr_scheduler.MultiStepLR(self.optimizer, self.schedule_milestone,
                                                        self.opt.lr_gamma))
        self.criterion = nn.L1Loss().to(self.device)
        self.criterion_weight = 1.0

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def get_data(self, data, need_H=True):
        self.L = data["L"].to(self.device)
        if need_H:
            self.H = data["H"].to(self.device)

    def model_run(self):
        self.pred_L = self.model(self.L)

    def optimize_parameters(self, current_step):
        self.optimizer.zero_grad()
        self.model_run()
        loss = self.criterion_weight * self.criterion(self.pred_L, self.H)
        loss.backward()
        self.optimizer.step()
        # optimizer clipgrad == none
        # regularizer orthstep == none
        # regularizer clipstep == none

        self.log_dict["loss"] = loss.item()

    def current_log(self):
        return self.log_dict

    def test(self):
        self.model.eval()
        with torch.no_grad():
            self.model_run()
        self.model.train()

    def save(self, iter_label):
        self.save_network(self.save_dir, self.model, "Model", iter_label)
        ### e_decay
        self.save_optimizer(self.save_dir, self.optimizer, "optimzierModel", iter_label)

    ###########################################################################

    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict["L"] = self.L.detach()[0].float().cpu()
        out_dict["E"] = self.pred_L.detach()[0].float().cpu()
        if need_H:
            out_dict["H"] = self.H.detach()[0].float().cpu()
        return out_dict

    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict["L"]=self.L.detach().float().cpu()
        out_dict["E"]=self.E.detach().float().cpu()
        if need_H:
            out_dict["H"] = self.H.detach().float().cpu()
        return out_dict

    ############################################################################

    def get_bare_model(self, network):
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def describe_network(self):
        network = self.model
        msg = "\n"
        msg += "Networks name: {}".format(network.__class__.__name__) + "\n"
        msg += "Params number: {}".format(sum(map(lambda x:x.numel(), network.parameters()))) + "\n"
        msg += "Net Structure:\n{}".format(str(network)) + "\n"
        return msg

    #########################################
    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self,model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    ##########################################
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        torch.save(state_dict, save_path)

    def save_optimizer(selfself, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = "{}_{}.pth".format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    ######################################################################################

    def load_network(self, load_path, network, strict=True, param_key="params"):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old,param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    #########################################################################################
    # to GAN Network useful training way

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)
