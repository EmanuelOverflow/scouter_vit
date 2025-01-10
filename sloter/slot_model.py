import torch
import torch.nn as nn
import torch.nn.functional as F
from sloter.utils.slot_attention import SlotAttention
from sloter.utils.position_encode import build_position_encoding
from timm.models import create_model
from collections import OrderedDict


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, num_channels, height, width)

    return x


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_classes)
    if args.dataset == "MNIST" and ('vit' not in args.model or 'transformer' not in args.model):
        bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
    if args.use_slot:
        if args.use_pre:
            checkpoint = torch.load(f"saved_model/{args.dataset}_no_slot_checkpoint.pth")
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model"].items():
                name = k[9:] # remove `backbone.`
                new_state_dict[name] = v
            bone.load_state_dict(new_state_dict)
            print("load pre dataset parameter over")
        if not args.grad:
            '''
            if 'seresnet' in args.model:
                bone.avg_pool = Identical()
                bone.last_linear = Identical()
            elif 'res' in args.model:
                bone.global_pool = Identical()
                bone.fc = Identical()
            elif 'efficient' in args.model:
                bone.global_pool = Identical()
                bone.classifier = Identical()
            elif 'densenet' in args.model:
                bone.global_pool = Identical()
                bone.classifier = Identical()
            elif 'mobilenet' in args.model:
                bone.global_pool = Identical()
                bone.conv_head = Identical()
                bone.act2 = Identical()
                bone.classifier = Identical()
            elif 'vit_base_patch' in args.model:
                
                bone.pool = Identical()
                bone.fc_norm = Identical()
                bone.head_drop = Identical()
                bone.head = Identical()
                
                bone.reset_classifier(0)  
                # Probably is sufficient to change bone.forward = bone.forward_features
            '''
            bone.reset_classifier(0) # new timm method
    return bone


class SlotModel(nn.Module):
    def __init__(self, args):
        super(SlotModel, self).__init__()
        self.use_slot = args.use_slot
        self.backbone = load_backbone(args)
        self.model_name = args.model
        
        self.channel = args.channel
        
        if self.use_slot:
            if 'densenet' in args.model:
                self.feature_size = 8
            else:
                self.feature_size = 9
            
            if 'vit' in self.model_name or 'transformer' in self.model_name:    
                self.channel_shuffle = args.channel_shuffle
                channels_out = args.channels_out
                rand_x = torch.rand(args.batch_size, channels_out)
                self.channel_shuffle_in = self.channels_rolling(rand_x).shape[1]
            else:
                self.channel_shuffle_in = args.channel    

            self.slots_per_class = args.slots_per_class
            self.conv1x1 = nn.Conv2d(self.channel_shuffle_in, args.hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            if args.pre_trained:
                self.dfs_freeze(self.backbone, args.freeze_layers)
            self.slot = SlotAttention(args.num_classes, self.slots_per_class, args.hidden_dim, vis=args.vis,
                                         vis_id=args.vis_id, loss_status=args.loss_status, power=args.power, to_k_layer=args.to_k_layer)
            self.position_emb = build_position_encoding('sine', hidden_dim=args.hidden_dim)
            self.lambda_value = float(args.lambda_value)
        else:
            if args.pre_trained:
                self.dfs_freeze(self.backbone, args.freeze_layers)

    def dfs_freeze(self, model, freeze_layer_num):
        if freeze_layer_num == 0:
            return

        unfreeze_layers = ['layer4', 'layer3', 'layer2', 'layer1'][:4-freeze_layer_num]
        for name, child in model.named_children():
            skip = False
            for freeze_layer in unfreeze_layers:
                if freeze_layer in name:
                    skip = True
                    break
            if skip:
                continue
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child, freeze_layer_num)

    def dfs_freeze_bnorm(self, model):
        for name, child in model.named_children():
            if 'bn' not in name:
                self.dfs_freeze_bnorm(child)
                continue
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze_bnorm(child)
            
    def channels_rolling(self, x):
        b = x.size(0)
        rolling_x = [x.view(b, self.channel, self.feature_size, self.feature_size)]
        equals = False
        x_initial = x.view(b, -1, 1, 1)
        shuffle_x = x_initial
        while not equals:
            shuffle_x = channel_shuffle(shuffle_x, self.channel_shuffle)
            if torch.all(shuffle_x == x_initial):
                equals = True
            else:
                rolling_x.append(shuffle_x.view(b, self.channel, self.feature_size, self.feature_size))
        return torch.concatenate(rolling_x, dim=1)

    def forward(self, x, target=None):
        x = self.backbone(x)
        if self.use_slot:
            if 'vit' in self.model_name or 'transformer' in self.model_name:    
                x = self.channels_rolling(x)
            x = self.conv1x1(x.view(x.size(0), self.channel_shuffle_in, self.feature_size, self.feature_size))
            x = torch.relu(x)
            pe = self.position_emb(x)
            x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            x, attn_loss = self.slot(x_pe, x)
        output = F.log_softmax(x, dim=1)

        if target is not None:
            if self.use_slot:
                loss = F.nll_loss(output, target) + self.lambda_value * attn_loss
                return [output, [loss, F.nll_loss(output, target), attn_loss]]
            else:
                loss = F.nll_loss(output, target)
                return [output, [loss]]

        return output
