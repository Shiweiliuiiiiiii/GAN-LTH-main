from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import math

def add_sparse_args(parser):
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--dy_mode', type=str, default='', help='dynamic change the sparse connectivity of which model')
    parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--G_growth', type=str, default='gradient', help='Growth mode for G. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--D_growth', type=str, default='random', help='Growth mode for D. Choose from: momentum, random, random_unfired, and gradient.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death_rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.5, help='The density level for balanced GAN')
    parser.add_argument('--update_frequency', type=int, default=2000, metavar='N', help='how many iterations to train between parameter exploration')
    parser.add_argument('--decay_schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--densityG', type=float, default=0.06, help='The density level of G for imbalanced GAN.')
    parser.add_argument('--densityD', type=float, default=0.06, help='The density level of D for imbalanced GAN.')
    parser.add_argument('--imbalanced', action='store_true', help='Enable balanced training mode. Default: True.')

def get_model_params(model):
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

def set_model_params(model, model_parameters):
    model.load_state_dict(model_parameters)

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


class Masking(object):
    def __init__(self, optimizer_G=False, optimizer_D_A=False, optimizer_D_B=False, death_rate_decay=False, death_rate=0.3, death='magnitude', G_growth='gradient', D_growth='random', redistribution='momentum', args=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if G_growth not in growth_modes:
            print('G_Growth mode: {0} not supported!'.format(G_growth))
            print('Supported modes are:', str(growth_modes))
        if D_growth not in growth_modes:
            print('D_Growth mode: {0} not supported!'.format(D_growth))
            print('Supported modes are:', str(growth_modes))

        self.device = torch.device("cuda")
        self.G_growth_mode = G_growth
        self.D_growth_mode = D_growth
        self.death_mode = death
        self.redistribution_mode = redistribution
        self.death_rate_decay = death_rate_decay
        self.args = args

        self.netG_A2B = None
        self.netG_B2A = None
        self.netD_A = None
        self.netD_B = None
        # masks
        self.netG_A2B_masks = {}
        self.netG_B2A_masks = {}
        self.netD_A_masks = {}
        self.netD_B_masks = {}
        # optimizers
        self.optimizer_G = optimizer_G
        self.optimizer_D_A = optimizer_D_A
        self.optimizer_D_B = optimizer_D_B
        # DST for which model
        self.dy_mode = args.dy_mode
        self.densityD = args.densityD
        self.densityG = args.densityG
        # stats
        self.death_rate = death_rate
        self.global_steps = 0

        # if fix, we do not explore the sparse connectivity
        if not self.dy_mode: self.prune_every_k_steps = None
        else: self.prune_every_k_steps = args.update_frequency

    def init(self, mode='ERK', density=0.05, densityG=0.5, erk_power_scale=1.0):
        # calculating density for G and D
        # netG_A2B, netG_B2A, netD_A, netD_B

        netG_A2B_total_params = 0
        for name, weight in self.netG_A2B_masks.items():
            netG_A2B_total_params += weight.numel()

        netG_B2A_total_params = 0
        for name, weight in self.netG_B2A_masks.items():
            netG_B2A_total_params += weight.numel()

        netD_A_total_params = 0
        for name, weight in self.netD_A_masks.items():
            netD_A_total_params += weight.numel()

        netD_B_total_params = 0
        for name, weight in self.netD_B_masks.items():
            netD_B_total_params += weight.numel()

        total_params = netG_A2B_total_params + netG_B2A_total_params + netD_A_total_params + netD_B_total_params

        if not self.args.imbalanced:
            self.G_density = density
            self.D_density = density
        else:
            self.G_density = self.densityG
            self.D_density = self.densityD

        print(f'Density of G is expected to be {self.G_density}')
        print(f'Density of D is expected to be {self.D_density}')


        if mode == 'ERK':
            print('initialize by ERK')
            print('initialize netG_A2B...')
            self.ERK_initialize(self.netG_A2B_masks, erk_power_scale, density=self.G_density)
            print('initialize netG_B2A...')
            self.ERK_initialize(self.netG_B2A_masks, erk_power_scale, density=self.G_density)
            print('initialize netD_A...')
            self.ERK_initialize(self.netD_A_masks, erk_power_scale, density=self.D_density)
            print('initialize netD_B...')
            self.ERK_initialize(self.netD_B_masks, erk_power_scale, density=self.D_density)


        self.apply_mask()

        self.netG_A2B_fired_masks = copy.deepcopy(self.netG_A2B_masks) # used for ITOP
        self.netG_B2A_fired_masks = copy.deepcopy(self.netG_B2A_masks) # used for ITOP
        self.netD_A_fired_masks = copy.deepcopy(self.netD_A_masks) # used for ITOP
        self.netD_B_fired_masks = copy.deepcopy(self.netD_B_masks) # used for ITOP

        print('---------------density of model netG_A2B------------------')
        for name, tensor in self.netG_A2B.named_parameters():
            print(name, (tensor!=0).sum().item()/tensor.numel())
        print('---------------density of model netG_B2A------------------')
        for name, tensor in self.netG_B2A.named_parameters():
            print(name, (tensor!=0).sum().item()/tensor.numel())
        print('---------------density of model netD_A------------------')
        for name, tensor in self.netD_A.named_parameters():
            print(name, (tensor!=0).sum().item()/tensor.numel())
        print('---------------density of model netD_B------------------')
        for name, tensor in self.netD_B.named_parameters():
            print(name, (tensor!=0).sum().item()/tensor.numel())


        netG_A2B_total_size = 0
        netG_A2B_total_size_sparse = 0
        for name, weight in self.netG_A2B_masks.items():
            netG_A2B_total_size += weight.numel()
            netG_A2B_total_size_sparse += (weight != 0).sum().int().item()
        print('Total Model parameters of netG_A2B:', netG_A2B_total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.G_density, netG_A2B_total_size_sparse / netG_A2B_total_size))

        netG_B2A_total_size = 0
        netG_B2A_total_size_sparse = 0
        for name, weight in self.netG_A2B_masks.items():
            netG_B2A_total_size += weight.numel()
            netG_B2A_total_size_sparse += (weight != 0).sum().int().item()
        print('Total Model parameters of netG_B2A:', netG_B2A_total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.G_density, netG_B2A_total_size_sparse / netG_B2A_total_size))

        nnetD_A_total_size = 0
        netD_A_total_size_sparse = 0
        for name, weight in self.netD_A_masks.items():
            nnetD_A_total_size += weight.numel()
            netD_A_total_size_sparse += (weight != 0).sum().int().item()
        print('Total Model parameters of netD_A:', nnetD_A_total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.D_density, netD_A_total_size_sparse / nnetD_A_total_size))

        nnetD_B_total_size = 0
        netD_B_total_size_sparse = 0
        for name, weight in self.netD_B_masks.items():
            nnetD_B_total_size += weight.numel()
            netD_B_total_size_sparse += (weight != 0).sum().int().item()
        print('Total Model parameters of netD_B:', nnetD_A_total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.D_density, netD_B_total_size_sparse / nnetD_B_total_size))


    def step(self, update_model):
        if 'G' in update_model:
            self.optimizer_G.step()
            self.apply_mask()

        if 'DA' in update_model:
            self.optimizer_D_A.step()
            self.apply_mask()

        if 'DB' in update_model:
            self.optimizer_D_B.step()
            self.apply_mask()
            # DST decay
            self.death_rate_decay.step()
            self.death_rate = self.death_rate_decay.get_dr()
            self.global_steps += 1

        if self.prune_every_k_steps is not None:
            # code for explore to do
            if self.global_steps % self.prune_every_k_steps == 0 and self.global_steps > 0 and 'DB' in update_model:
                if 'G' in self.dy_mode:
                    # parameter exploration for netG_A2B
                    self.weight_exploration(self.netG_A2B, self.netG_A2B_masks)
                    # parameter exploration for netG_B2A
                    self.weight_exploration(self.netG_B2A, self.netG_B2A_masks)
                elif 'D' in self.dy_mode:
                    # parameter exploration for netD_A
                    self.weight_exploration(self.netD_A, self.netD_A_masks)

                    # parameter exploration for netD_B
                    self.weight_exploration(self.netD_B, self.netD_B_masks)

                self.print_nonzero_counts()
                self.fired_masks_update()

    def add_module(self, netG_A2B, netG_B2A, netD_A, netD_B, densityG=0.5 , density=0.5, sparse_init='ERK'):
        self.netG_A2B = netG_A2B
        self.netG_B2A = netG_B2A
        self.netD_A = netD_A
        self.netD_B = netD_B

        # add masks for netG_A2B
        for name, tensor in self.netG_A2B.named_parameters():
            if len(tensor.size()) == 4:
                self.netG_A2B_masks[name] = torch.zeros_like(tensor,  requires_grad=False).cuda()
        # add masks for netG_B2A
        for name, tensor in self.netG_B2A.named_parameters():
            if len(tensor.size()) == 4:
                self.netG_B2A_masks[name] = torch.zeros_like(tensor,  requires_grad=False).cuda()
        # add masks for netD_A
        for name, tensor in self.netD_A.named_parameters():
            if len(tensor.size()) == 4:
                self.netD_A_masks[name] = torch.zeros_like(tensor,  requires_grad=False).cuda()
        # add masks for netD_B
        for name, tensor in self.netD_B.named_parameters():
            if len(tensor.size()) == 4:
                self.netD_B_masks[name] = torch.zeros_like(tensor,  requires_grad=False).cuda()

        # initialize sparse models
        self.init(mode=sparse_init, density=density, densityG=densityG)

    def apply_mask(self):

        # apply masks to netG_A2B
        # extract sparse connectivity
        model_para = get_model_params(self.netG_A2B)
        # change sparse connectivity
        for name in model_para:
            if name in self.netG_A2B_masks:
                model_para[name] = model_para[name]*self.netG_A2B_masks[name]
        # apply masks
        set_model_params(self.netG_A2B, model_para)

        # apply masks to netG_B2A
        model_para = get_model_params(self.netG_B2A)
        for name in model_para:
            if name in self.netG_B2A_masks:
                model_para[name] = model_para[name] * self.netG_B2A_masks[name]
        set_model_params(self.netG_B2A, model_para)

        model_para = get_model_params(self.netD_A)
        for name in model_para:
            if name in self.netD_A_masks:
                model_para[name] = model_para[name] * self.netD_A_masks[name]
        set_model_params(self.netD_A, model_para)

        model_para = get_model_params(self.netD_B)
        for name in model_para:
            if name in self.netD_B_masks:
                model_para[name] = model_para[name] * self.netD_B_masks[name]
        set_model_params(self.netD_B, model_para)

    def weight_exploration(self, model, masks):
        self.name2nonzeros = {}
        self.name2zeros = {}
        for name, weight in model.named_parameters():
            if name not in masks: continue
            mask = masks[name]
            self.name2nonzeros[name] = mask.sum().item()
            self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
            new_mask = self.magnitude_death(mask, weight, name, self.name2nonzeros[name], self.name2zeros[name])
            masks[name][:] = new_mask
        # grow
        for name, weight in model.named_parameters():
            if name not in masks: continue
            new_mask = masks[name].data.byte()
            num_remove = int(self.name2nonzeros[name] - new_mask.sum().item())

            # growth
            if self.G_growth_mode == 'random':
                new_mask = self.random_growth(name, new_mask, weight, num_remove)

            if self.G_growth_mode == 'random_unfired':
                new_mask = self.random_unfired_growth(name, new_mask, weight, num_remove)

            elif self.G_growth_mode == 'momentum':
                new_mask = self.momentum_growth(name, new_mask, weight, num_remove)

            elif self.G_growth_mode == 'gradient':
                new_mask = self.gradient_growth(name, new_mask, weight, num_remove)

            # exchanging masks
            masks.pop(name)
            masks[name] = new_mask.float()

        self.apply_mask()
    '''
                    DEATH
    '''
    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.death_rate * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name, num_nonzero, num_zero):

        num_remove = math.ceil(self.death_rate*num_nonzero)
        if num_remove == 0.0: return weight.data != 0.0

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zero + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.death_rate*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_unfired_growth(self, name, new_mask, weight, num_remove):
        total_regrowth = num_remove
        n = (new_mask == 0).sum().item()
        if n == 0: return new_mask
        num_nonfired_weights = (self.fired_masks[name]==0).sum().item()

        if total_regrowth <= num_nonfired_weights:
            idx = (self.fired_masks[name].flatten() == 0).nonzero()
            indices = torch.randperm(len(idx))[:total_regrowth]

            # idx = torch.nonzero(self.fired_masks[name].flatten())
            new_mask.data.view(-1)[idx[indices]] = 1.0
        else:
            new_mask[self.fired_masks[name]==0] = 1.0
            n = (new_mask == 0).sum().item()
            expeced_growth_probability = ((total_regrowth-num_nonfired_weights) / n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
            new_mask = new_mask.byte() | new_weights
        return new_mask

    def random_growth(self, name, new_mask, weight, num_remove):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (num_remove/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask_ = new_mask.bool() | new_weights
        if (new_mask_!=0).sum().item() == 0:
            new_mask_ = new_mask
        return new_mask_

    def momentum_growth(self, name, new_mask, weight, num_remove):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:num_remove]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, weight, num_remove):
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:num_remove]] = 1.0

        return new_mask



    def momentum_neuron_growth(self, name, new_mask, weight, num_remove):
        total_regrowth = self.num_remove[name]
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''

    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        print(f'-------------------------netG_A2B------------------------------------')
        for name, tensor in self.netG_A2B.named_parameters():
            if name not in self.netG_A2B_masks: continue
            mask = self.netG_A2B_masks[name]
            num_nonzeros = (mask != 0).sum().item()
            val = '{0}: {1}, density: {2:.3f}'.format(name, num_nonzeros, num_nonzeros/float(mask.numel()))
            print(val)
        print(f'-------------------------netG_B2A------------------------------------')
        for name, tensor in self.netG_B2A.named_parameters():
            if name not in self.netG_B2A_masks: continue
            mask = self.netG_B2A_masks[name]
            num_nonzeros = (mask != 0).sum().item()
            val = '{0}: {1}, density: {2:.3f}'.format(name, num_nonzeros, num_nonzeros/float(mask.numel()))
            print(val)
        print(f'-------------------------netD_A------------------------------------')
        for name, tensor in self.netD_A.named_parameters():
            if name not in self.netD_A_masks: continue
            mask = self.netD_A_masks[name]
            num_nonzeros = (mask != 0).sum().item()
            val = '{0}: {1}, density: {2:.3f}'.format(name, num_nonzeros, num_nonzeros/float(mask.numel()))
            print(val)
        print(f'-------------------------netD_B------------------------------------')
        for name, tensor in self.netD_B.named_parameters():
            if name not in self.netD_B_masks: continue
            mask = self.netD_B_masks[name]
            num_nonzeros = (mask != 0).sum().item()
            val = '{0}: {1}, density: {2:.3f}'.format(name, num_nonzeros, num_nonzeros/float(mask.numel()))
            print(val)

        print('Death rate: {0}\n'.format(self.death_rate))

    def fired_masks_update(self):
        netG_A2B_ntotal_fired_weights = 0.0
        netG_A2B_ntotal_weights = 0.0
        netG_A2B_layer_fired_weights = {}
        for name, weight in self.netG_A2B.named_parameters():
            if name not in self.netG_A2B_masks: continue
            self.netG_A2B_fired_masks[name] = self.netG_A2B_masks[name].data.byte() | self.netG_A2B_fired_masks[name].data.byte()
            netG_A2B_ntotal_fired_weights += float(self.netG_A2B_fired_masks[name].sum().item())
            netG_A2B_ntotal_weights += float(self.netG_A2B_fired_masks[name].numel())
            netG_A2B_layer_fired_weights[name] = float(self.netG_A2B_fired_masks[name].sum().item()) / float(
                self.netG_A2B_fired_masks[name].numel())
            print('Layerwise percentage of the fired weights of', name, 'is:', netG_A2B_layer_fired_weights[name])
        netG_A2B_total_fired_weights = netG_A2B_ntotal_fired_weights / netG_A2B_ntotal_weights
        print('The percentage of the total fired weights of netG_A2B is:', netG_A2B_total_fired_weights)
        print(f'-------------------------netD_B------------------------------------')
        netG_B2A_ntotal_fired_weights = 0.0
        netG_B2A_ntotal_weights = 0.0
        netG_B2A_layer_fired_weights = {}
        for name, weight in self.netG_B2A.named_parameters():
            if name not in self.netG_B2A_masks: continue
            self.netG_B2A_fired_masks[name] = self.netG_B2A_masks[name].data.byte() | self.netG_B2A_fired_masks[
                name].data.byte()
            netG_B2A_ntotal_fired_weights += float(self.netG_B2A_fired_masks[name].sum().item())
            netG_B2A_ntotal_weights += float(self.netG_B2A_fired_masks[name].numel())
            netG_B2A_layer_fired_weights[name] = float(self.netG_B2A_fired_masks[name].sum().item()) / float(
                self.netG_B2A_fired_masks[name].numel())
            print('Layerwise percentage of the fired weights of', name, 'is:', netG_B2A_layer_fired_weights[name])
        netG_B2A_total_fired_weights = netG_B2A_ntotal_fired_weights / netG_B2A_ntotal_weights
        print('The percentage of the total fired weights of netG_B2A is:', netG_B2A_total_fired_weights)
        print(f'-------------------------netD_B------------------------------------')
        netD_A_ntotal_fired_weights = 0.0
        netD_A_ntotal_weights = 0.0
        netD_A_layer_fired_weights = {}
        for name, weight in self.netD_A.named_parameters():
            if name not in self.netD_A_masks: continue
            self.netD_A_fired_masks[name] = self.netD_A_masks[name].data.byte() | self.netD_A_fired_masks[
                name].data.byte()
            netD_A_ntotal_fired_weights += float(self.netD_A_fired_masks[name].sum().item())
            netD_A_ntotal_weights += float(self.netD_A_fired_masks[name].numel())
            netD_A_layer_fired_weights[name] = float(self.netD_A_fired_masks[name].sum().item()) / float(
                self.netD_A_fired_masks[name].numel())
            print('Layerwise percentage of the fired weights of', name, 'is:', netD_A_layer_fired_weights[name])
        netD_A_total_fired_weights = netD_A_ntotal_fired_weights / netD_A_ntotal_weights
        print('The percentage of the total fired weights of etD_A is:', netD_A_total_fired_weights)
        print(f'-------------------------netD_B------------------------------------')
        netD_B_ntotal_fired_weights = 0.0
        netD_B_ntotal_weights = 0.0
        netD_B_layer_fired_weights = {}
        for name, weight in self.netD_B.named_parameters():
            if name not in self.netD_B_masks: continue
            self.netD_B_fired_masks[name] = self.netD_B_masks[name].data.byte() | self.netD_B_fired_masks[
                name].data.byte()
            netD_B_ntotal_fired_weights += float(self.netD_B_fired_masks[name].sum().item())
            netD_B_ntotal_weights += float(self.netD_B_fired_masks[name].numel())
            netD_B_layer_fired_weights[name] = float(self.netD_B_fired_masks[name].sum().item()) / float(
                self.netD_B_fired_masks[name].numel())
            print('Layerwise percentage of the fired weights of', name, 'is:', netD_B_layer_fired_weights[name])
        netD_B_total_fired_weights = netD_B_ntotal_fired_weights / netD_B_ntotal_weights
        print('The percentage of the total fired weights of etD_B is:', netD_B_total_fired_weights)
        print(f'-------------------------netD_B------------------------------------')
    def ERK_initialize(self, masks ,erk_power_scale, density):
        total_params = 0
        for name, weight in masks.items():
            total_params += weight.numel()
        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in masks.items():
                n_param = np.prod(mask.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(mask.shape) / np.prod(mask.shape)
                                              ) ** erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in masks.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
            )
            masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall sparsity of  {total_nonzero / total_params}")


