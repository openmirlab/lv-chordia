import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim  # Still needed for checkpoint loading (optimizer state)
from ..common import WORKING_PATH
import os
import numpy as np
from typing import Optional

class NetworkBehavior(nn.Module):

    def __init__(self, use_gpu: Optional[bool]=None):
        super().__init__()
        # use_gpu=None (the default) preserves the original auto-detect
        # behavior exactly; callers that want an explicit override (e.g. to
        # force CPU on a CUDA-capable machine) pass True/False.
        self.use_gpu=torch.cuda.device_count()>0 if use_gpu is None else use_gpu
        self.use_data_parallel=False

    def get_optimizer(self):
        return optim.Adam(self.parameters())

    def forward(self, *args):
        raise NotImplementedError()

    def init_settings(self, is_training):
        if(self.use_gpu):
            self.cuda()
        else:
            self.cpu()
        if(is_training):
            self.train()
        else:
            self.eval()
        if(self.use_data_parallel):
            self.parallel_net=[nn.DataParallel(self)]

    def feed(self, *args):
        if(self.use_data_parallel):
            return self.parallel_net[0](*args)
        else:
            return self(*args)

    def loss(self, *args):
        raise NotImplementedError()

    def inference(self, *args):
        raise NotImplementedError()

    def evaluation(self, *args):
        raise NotImplementedError()

class NetworkInterface:

    def __init__(self, net, save_name, load_checkpoint=False, load_path='cache_data'):
        from ..common import CACHE_DATA_PATH
        self.net=net
        if(not isinstance(self.net,NetworkBehavior)):
            raise Exception('Invalid network type')
        if('(p)' in save_name):
            self.net.use_data_parallel=True
        self.net.init_settings(False)
        self.save_name=save_name
        # Use CACHE_DATA_PATH if load_path is 'cache_data', otherwise use WORKING_PATH
        self.base_path = CACHE_DATA_PATH if load_path == 'cache_data' else os.path.join(WORKING_PATH, load_path)
        save_path=os.path.join(self.base_path,'%s.sdict'%save_name)
        cp_save_path=os.path.join(self.base_path,'%s.cp.sdict'%save_name)
        self.finalized=False
        self.optimizer=self.net.get_optimizer()
        self.counter=0
        self.best_val_loss=np.inf
        self.best_epoch_dist=0
        if(os.path.exists(save_path)):
            state_dict=torch.load(save_path,map_location='cuda' if self.net.use_gpu else 'cpu')
            # The following codes are for torch 4.0 compatibility
            # new_state_dict={}
            # for key in state_dict['net']:
            #     if('num_batches_tracked' not in key):
            #         new_state_dict[key]=state_dict['net'][key]
            # self.net.load_state_dict(new_state_dict)
            self.net.load_state_dict(state_dict['net'])
            self.counter=state_dict['counter']
            self.optimizer.load_state_dict(state_dict['opt'])
            try:
                self.best_epoch_dist=state_dict['best_epoch_dist']
                self.best_val_loss=state_dict['best_val_loss']
            except:
                pass
            self.finalized=True
        elif(load_checkpoint and os.path.exists(cp_save_path)):
            state_dict=torch.load(cp_save_path,map_location='cuda' if self.net.use_gpu else 'cpu')
            # The following codes are for torch 4.0 compatibility
            # new_state_dict={}
            # for key in state_dict['net']:
            #     if('num_batches_tracked' not in key):
            #         new_state_dict[key]=state_dict['net'][key]
            # self.net.load_state_dict(new_state_dict)
            self.net.load_state_dict(state_dict['net'])
            self.counter=state_dict['counter']
            self.optimizer.load_state_dict(state_dict['opt'])
            try:
                self.best_epoch_dist=state_dict['best_epoch_dist']
                self.best_val_loss=state_dict['best_val_loss']
            except:
                pass

    # Removed train_supervised() method - training-only, not needed for inference-only package

    def inference(self, *args,**kwargs):
        self.net.init_settings(False)
        inputs=[torch.tensor(arg,dtype=torch.float if arg.dtype in [np.float16,np.float32,np.float64] else torch.long)
                for arg in args]
        if(self.net.use_gpu):
            inputs=[input.cuda() for input in inputs]
        with torch.no_grad():
            return self.net.inference(*inputs,**kwargs)

    def inference_function(self,function,*args,**kwargs):
        self.net.init_settings(False)
        inputs=[torch.tensor(arg,dtype=torch.float if arg.dtype in [np.float16,np.float32,np.float64] else torch.long)
                for arg in args]
        if(self.net.use_gpu):
            inputs=[input.cuda() for input in inputs]
        with torch.no_grad():
            return self.net.__class__.__dict__[function](self.net,*inputs,**kwargs)


