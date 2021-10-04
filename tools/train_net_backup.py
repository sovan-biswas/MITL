#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import copy
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import time
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model,moco_wrapper
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, EpochTimer
from slowfast.utils.multigrid import MultigridSchedule
from torch import bincount
logger = logging.get_logger(__name__)
class Queue_data(torch.nn.Module):
    def __init__(self, K=256, num_classes =80, cfg=None):
        super(Queue_data, self).__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE if cfg != None else 224
        self.num_channels = 3
        self.K = K
        self.num_frames_slow = 8
        self.num_frames_fast = 32
        self.num_classes = num_classes
        self.max_boxes = 5

        # self.register_buffer("queue_frames_slow", torch.zeros(self.K, self.num_channels, self.num_frames_slow,self.img_size,self.img_size,dtype=torch.half))
        self.register_buffer("queue_frames_fast",torch.zeros(self.K, self.num_channels, self.num_frames_fast, self.img_size, self.img_size,dtype=torch.half,requires_grad=False).cpu())
        self.register_buffer("queue_locs",torch.zeros(self.K, self.max_boxes, 5, dtype=torch.half,requires_grad=False))
        # self.register_buffer("queue_labels", torch.zeros(self.K, self.max_boxes, self.num_classes,dtype=torch.bool))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long,requires_grad=False))
        self.register_buffer("queue_samples", torch.zeros(1, dtype=torch.long,requires_grad=False))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, inputs, locs):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)


        batch_size = inputs.shape[0]
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity
        #
        # # replace the keys at ptr (dequeue and enqueue)
        # self.queue_frames_slow[ptr:ptr + batch_size,:,:,:,:] = inputs[0].half()
        self.queue_frames_fast[ptr:ptr + batch_size, :, :, :, :] = inputs.half()
        cnts = torch.bincount(locs[:, 0].type(torch.int))
        locs[:, 0] += ptr
        split_locs = locs.split(tuple(cnts.tolist()), dim=0)
        locs = torch.nn.utils.rnn.pad_sequence(split_locs, batch_first=True)
        n_box = locs.shape[1]
        self.queue_locs[ptr:ptr + batch_size, :n_box, :] = locs[:,:n_box,:].half()
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr
        self.queue_samples[0] = min(int(self.queue_samples + batch_size),int(self.K))

    @torch.no_grad()
    def forward(self, inputs, locs):
        self._dequeue_and_enqueue(inputs,locs)
        samples = int(self.queue_samples)
        return self.queue_frames_fast[:samples], self.queue_locs[:samples]

def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer=None, dictionary_queue = None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, idxs, meta) in enumerate(train_loader):
        input_cpu = inputs[1]
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        # #### diambiguate orig data from augmentation
        # im_q = [inputs[0][0::2, :, :, :, :], inputs[1][0::2, :, :, :, :]]
        # im_k = [inputs[0][1::2, :, :, :, :], inputs[1][1::2, :, :, :, :]]
        # boxes_q = boxes[boxes[:, 0] % 2 == 0, :]
        # boxes_q[:, 0] = boxes_q[:, 0] // 2
        # boxes_k = boxes[boxes[:, 0] % 2 == 1, :]
        # boxes_k[:, 0] = boxes_k[:, 0] // 2
        # boxes_q = boxes[boxes[:, 0] % 2 == 0, :]
        # boxes_q[:, 0] = boxes_q[:, 0] // 2

        if cfg.DETECTION.ENABLE:
            ### Original image
            orig_inputs = [inputs[0][0::3, :, :, :, :], inputs[1][0::3, :, :, :, :]]
            # _, counts_box = meta['boxes'][:, 0].unique(return_counts=True)
            counts_box = torch.bincount(meta['boxes'][:, 0].type(torch.int))
            meta_copy_boxes = meta["boxes"].split(tuple(counts_box.tolist()), dim=0)
            meta_copy_ori_boxes = meta["ori_boxes"].split(tuple(counts_box.tolist()), dim=0)
            labels_copy = labels.split(tuple(counts_box.tolist()), dim=0)
            meta_boxes = torch.cat(meta_copy_boxes[0::3],dim=0)
            meta_boxes[:, 0] = meta_boxes[:, 0] // 3.0
            meta_boxes_ori = torch.cat(meta_copy_ori_boxes[0::3], dim=0)
            meta_boxes_ori[:, 0] = meta_boxes_ori[:, 0] // 3.0
            bag_indx = meta_boxes[:, 0].clone()
            # preds, feat = model(orig_inputs,meta_boxes)
            labels = torch.cat(labels_copy[0::3],dim=0)
            counts = counts_box[0::3]
            bag_label = misc.convert_into_bag(labels, bag_indx, requires_grad=False, counts=counts)
            meta_boxes_ori = misc.convert_into_bag(meta_boxes_ori, bag_indx, requires_grad=False, counts=counts)
            meta_boxes_bag = misc.convert_into_bag(meta_boxes.clone(), bag_indx, requires_grad=False, counts=counts)

            pos_inputs = [inputs[0][1::3, :, :, :, :], inputs[1][1::3, :, :, :, :]]
            pos_meta_boxes = torch.cat(meta_copy_boxes[1::3], dim=0)
            pos_meta_boxes[:,0] = pos_meta_boxes[:,0]//3
            pos_meta_boxes_ori = torch.cat(meta_copy_ori_boxes[1::3], dim=0)
            pos_meta_boxes_ori[:, 0] = pos_meta_boxes_ori[:, 0] // 3.0
            pos_bag_indx = pos_meta_boxes[:, 0].clone()
            pos_labels = torch.cat(labels_copy[1::3], dim=0)
            pos_counts = counts_box[1::3]
            pos_labels = misc.convert_into_bag(pos_labels, pos_bag_indx, requires_grad=False,counts=pos_counts)
            pos_meta_boxes_ori = misc.convert_into_bag(pos_meta_boxes_ori, pos_bag_indx, requires_grad=False, counts=pos_counts)

            neg_inputs = [inputs[0][2::3, :, :, :, :], inputs[1][2::3, :, :, :, :]]
            neg_meta_boxes = torch.cat(meta_copy_boxes[2::3], dim=0)
            neg_meta_boxes[:, 0] = neg_meta_boxes[:, 0] // 3
            neg_labels = torch.cat(labels_copy[2::3], dim=0)
            neg_bag_indx = neg_meta_boxes[:, 0].clone()
            neg_counts = counts_box[2::3]
            neg_labels = misc.convert_into_bag(neg_labels, neg_bag_indx, requires_grad=False, counts=neg_counts)

            del meta_copy_boxes, meta_copy_ori_boxes, labels_copy
            #### Get the dictionary
            if dictionary_queue is not None:
                _ = dictionary_queue(input_cpu[1::3, :, :, :, :],pos_meta_boxes)
                # dict_preds_pos, dict_feat_pos = copy.deepcopy(model).half()(d_inp_p, d_locs_p)
                # d_inp_fast, d_locs = dictionary_queue(input_cpu[2::3, :, :, :, :],neg_meta_boxes)
                _ = dictionary_queue(input_cpu[2::3, :, :, :, :],neg_meta_boxes)
                # if cur_epoch >= 1 and (cur_iter + 1) % 4 == 0:
                #     d_ = []
                #     for i, d in enumerate(zip(d_inp_fast.clone().split(16, dim=0), d_locs.clone().split(16, dim=0))):
                #         d[-1][:,:,0] -= i*16
                #         d_valid = d[-1][:,:,1:].sum(-1) > 0
                #         fr_indx = torch.linspace(0, d[0].shape[2] - 1, d[0].shape[2] // cfg.SLOWFAST.ALPHA).long()
                #         img_fast = d[0].cuda().float()
                #         img_slow = img_fast[:,:,fr_indx,:,:]
                #         d_.append(model([img_slow, img_fast], d[-1][d_valid].float()))
                #     d1, d2 = zip(*d_)
                #     dict_preds, dict_feat = torch.cat(d1,dim=0),torch.cat(d2,dim=0)
                #     del d1, d2, img_fast, img_slow, d_inp_fast, d_locs, d, d_
            with torch.no_grad():
                model.train(False)
                model.eval()
                neg_preds, neg_feat = model(neg_inputs, neg_meta_boxes)
                pos_preds, pos_feat = model(pos_inputs, pos_meta_boxes)
                pos_bag_feat = misc.convert_into_bag(pos_feat, pos_bag_indx, requires_grad=False, counts=pos_counts)
                neg_bag_feat = misc.convert_into_bag(neg_feat, neg_bag_indx, requires_grad=False, counts=neg_counts)
                model.train()
            preds, feat = model(orig_inputs, meta_boxes)
            # # preds, feat, uncertainity = model(inputs, meta["boxes"])
            # preds, feat = model(inputs, meta["boxes"])
        else:
            preds = model(inputs)
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="none")
        kl_loss_fun = torch.nn.KLDivLoss(reduction='mean')
        # loss_soft_plus = losses.get_loss_func("softplus")(beta=-1)
        # loss_fun_contrast = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # #### Divide into orig feat and negative feature
        # orig_instant, neg_instant, _ = misc.get_orig_pos_neg(preds, meta['boxes'][:, 0],
        #                                                                        meta['ori_boxes'][:, 0], cfg)
        # orig_feat, neg_feat, _ = misc.get_orig_pos_neg(feat, meta['boxes'][:, 0],
        #                                                      meta['ori_boxes'][:, 0], cfg)
        # # uncertainity, neg_uncertainity, _ = misc.get_orig_pos_neg(uncertainity, meta['boxes'][:, 0],
        # #                                                meta['ori_boxes'][:, 0], cfg)
        # orig_locs, neg_locs, _ = misc.get_orig_pos_neg(meta['boxes'], meta['boxes'][:, 0],
        #                                                          meta['ori_boxes'][:, 0], cfg)
        # label_weight, _, ninst = misc.get_orig_pos_neg(meta['scores'], meta['ori_boxes'][:, 0],
        #                                                   meta['ori_boxes'][:, 0])

        #### Compute Bag
        ### Label
        # counts = meta_boxes_ori[:, 0].unique(return_counts=True)[1]
        # pos_counts = pos_meta_boxes_ori[:, 0].unique(return_counts=True)[1]
        # neg_counts = neg_meta_boxes[:, 0].unique(return_counts=True)[1]
        # print("Sorting: ", time.time()-start)
        # start = time.time()



        ### Predictions
        bag_pred = misc.convert_into_bag(preds, bag_indx, counts=counts)
        # pos_bag_pred = misc.convert_into_bag(pos_preds, pos_meta_boxes[:, 0], requires_grad=False, counts=pos_counts)
        # neg_bag_pred = misc.convert_into_bag(neg_preds, neg_meta_boxes[:, 0], requires_grad=False, counts=neg_counts)

        ### Similarity head
        bag_feat = misc.convert_into_bag(feat, bag_indx, pad_val=0.0, counts=counts)

        ### Boxes Location

        # pos_meta_boxes = misc.convert_into_bag(pos_meta_boxes, pos_meta_boxes[:, 0], requires_grad=False, counts=pos_counts)
        # neg_meta_boxes = misc.convert_into_bag(neg_meta_boxes, neg_meta_boxes[:, 0], requires_grad=False, counts=neg_counts)

        ### Boxes Location


        #### Bag Prediction Loss
        bag_weight = torch.ones_like(bag_label.max(1)[0])
        bag_weight[bag_label.max(1)[0]==1.0] = 0.5
        loss = (loss_fun(bag_pred.max(1)[0], bag_label.max(1)[0]) * bag_weight).mean()

        # # uncertainity = -loss_soft_plus(uncertainity)
        # # bag_pred, uncertainity_score = misc.convert_into_bag(orig_instant, meta['ori_boxes'][:, 0],
        # #                                                      confidence=uncertainity)
        # bag_pred = misc.convert_into_bag(preds, meta_boxes[:, 0])
        # bag_feat = misc.convert_into_bag(feat, meta_boxes[:, 0])
        #
        # pos_bag_feat = misc.convert_into_bag(pos_feat, pos_meta_boxes[:, 0], requires_grad=False)
        # neg_bag_feat = misc.convert_into_bag(neg_feat, neg_meta_boxes[:, 0], requires_grad=False)

        #### Contrastive Loss
        orig_valid = (meta_boxes_bag[:, :, 3].sub(meta_boxes_bag[:, :, 1]) * meta_boxes_bag[:, :, 4].sub(meta_boxes_bag[:, :, 2]) > 1024)
        orig_feat_normalize = torch.nn.functional.normalize(bag_feat, dim=-1)

        # pos_valid = (pos_meta_boxes[:, :, 3].sub(pos_meta_boxes[:, :, 1]) * pos_meta_boxes[:, :, 4].sub(pos_meta_boxes[:, :, 2]) > 0)
        pos_feat_normalize = torch.nn.functional.normalize(pos_bag_feat, dim=-1)

        l1 = meta_boxes_ori[:, :, 1:]
        l2 = pos_meta_boxes_ori[:, :, 1:]
        o1 = torch.cat([l1[:, :, 0::2].mean(-1, True), l1[:, :, 1::2].mean(-1, True)], dim=-1)
        o2 = torch.cat([l2[:, :, 0::2].mean(-1, True), l2[:, :, 1::2].mean(-1, True)], dim=-1)
        weight = (-torch.cdist(o1, o2, p=2.0)).exp()

        inter_bag_best_match = (torch.einsum("ijk,ikl->ijl", orig_feat_normalize, pos_feat_normalize.permute(0, 2, 1)) * weight)
        # inter_bag_best_match = (torch.einsum("ijk,ikl->ijl", orig_feat_normalize, pos_feat_normalize.permute(0, 2, 1)))
        # inter_bag_best_match = inter_bag_best_match * pos_valid.unsqueeze(dim=1).type_as(inter_bag_best_match)
        inter_bag_best_match = inter_bag_best_match.amax(dim=(-2,-1))


        neg_feat_normalize = torch.nn.functional.normalize(neg_bag_feat, dim=-1)
        # neg_valid = (neg_meta_boxes[:, :, 3].sub(neg_meta_boxes[:, :, 1]) * neg_meta_boxes[:, :, 4].sub(neg_meta_boxes[:, :, 2]) > 0)
        inter_bag_neg_match = torch.einsum("ijk,ikl->ijl", orig_feat_normalize, neg_feat_normalize.permute(0, 2, 1))
        inter_bag_neg_match[inter_bag_neg_match==0.0] = 1.0
        inter_bag_neg_match = inter_bag_neg_match.amin(dim=(-2,-1))
        # inter_bag_neg_match = inter_bag_neg_match.amax(dim=(-2,-1))


        # bag_feat = torch.nn.functional.normalize(bag_feat.max(1)[0],dim=-1) ## orig view
        # pos_bag_feat = torch.nn.functional.normalize(pos_bag_feat.max(1)[0],dim=-1) #.detach() ## aug view
        # neg_bag_feat = torch.nn.functional.normalize(neg_bag_feat.max(1)[0],dim=-1) #.detach() ## neg view

        ### Bag Contrast
        # contrast_loss = (0.3 + torch.einsum("ij,ij->i", bag_feat, neg_bag_feat) - torch.einsum("ij,ij->i", bag_feat,pos_bag_feat)).clamp(min=0.0).mean()
        # inst_contrast_loss = (0.2 + inter_bag_neg_match - inter_bag_best_match).clamp(min=0.0).sum() / (len(inter_bag_neg_match) + 0.000001)
        # sim_loss = (0.8 - torch.einsum("ij,ij->i", bag_feat, pos_bag_feat)).clamp(min=0.0).mean() + 0.5 * (0.8 - inter_bag_best_match).clamp(min=0.0).sum() / (len(inter_bag_best_match) + 0.000001)
        # loss += (1.0 * contrast_loss + 0.5 * inst_contrast_loss + sim_loss)
        contrast_loss = (0.05 + inter_bag_neg_match - inter_bag_best_match).clamp(min=0.0).mean()
        # contrast_loss = (0.1 + inter_bag_neg_match - inter_bag_best_match).clamp(min=0.0).mean()
        sim_loss = (0.8 - inter_bag_best_match).clamp(min=0.0).mean()
        loss += (1.0 * contrast_loss + 1.0 * sim_loss)
        # contrast_loss = torch.log(1+((inter_bag_neg_match - inter_bag_best_match + 0.1)/0.07).exp()).mean()
        # sim_loss = torch.log(1+((-inter_bag_best_match)/0.07).exp()).mean()
        # loss += (1.0 * contrast_loss + 1.0 * sim_loss)

        # # loss += (inter_bag_best_match + inter_bag_neg_match + intra_bag_neg_match)
        #

        #### Learn from others
        # if dictionary_queue is None and pos_feat_normalize[pos_valid].shape[0] > 0 and orig_feat_normalize[orig_valid].shape[0] > 0:
        #     dictionary = torch.cat([pos_feat_normalize[pos_valid], neg_feat_normalize[neg_valid]], dim=0)
        #     dictionary_prob = torch.cat([pos_bag_pred[pos_valid], neg_bag_pred[neg_valid]], dim=0)
        #     labels_bag = torch.cat([pos_labels[pos_valid], neg_labels[neg_valid]], dim=0)
        #     sim = torch.einsum("ij,jk->ik", orig_feat_normalize[orig_valid], dictionary.permute(1, 0)) / 0.07
        #     sim = sim.clamp(min=-30,max=30).exp()
        #     sim = sim / sim.sum(-1, True)
        #     weighted_representation = torch.einsum('ij,ij->ij', bag_label[orig_valid],torch.einsum('ij,jk->ik', sim, (dictionary_prob * labels_bag)))
        #     rep = bag_pred[orig_valid]
        #     kl_loss = kl_loss_fun(rep.log(), weighted_representation) + kl_loss_fun((1 - rep).log(),
        #                                                                             1 - weighted_representation)
        #     loss += 1.0 * kl_loss
        if dictionary_queue is not None and cur_iter >= 1 and 'dict_feat' in locals():
            # dictionary = torch.nn.functional.normalize(torch.cat([dict_feat_pos,dict_feat_neg],dim=0).float(),dim=-1)
            # dictionary_prob = torch.cat([dict_preds_pos,dict_preds_neg],dim=0).float()
            # labels_bag = torch.cat([dict_labels_pos,dict_labels_neg],dim=0).float()
            dictionary = torch.nn.functional.normalize(dict_feat, dim=-1)
            dictionary_prob = dict_preds
            # labels_bag = dict_labels.float()
            sim = torch.einsum("ij,jk->ik", orig_feat_normalize[orig_valid], dictionary.permute(1, 0)) / 0.07
            sim = sim.exp()
            sim = sim / sim.sum(-1, True)
            weighted_representation = torch.einsum('ij,ij->ij', bag_label[orig_valid],
                                                   torch.einsum('ij,jk->ik', sim, dictionary_prob))
            rep = bag_pred[orig_valid]
            kl_loss = kl_loss_fun(rep.clamp(0.00001,.99999).log(), weighted_representation) + kl_loss_fun((1 - rep).clamp(0.00001,.99999).log(),
                                                                                    1 - weighted_representation)
            misc.check_nan_losses(kl_loss)
            loss += 1.0 * kl_loss

        # i1 = ii[0::3, :, :]
        # i2 = ii[1::3, :, :]
        # i3 = ii[2::3, :, :]
        #
        # lab1 = torch.nn.utils.rnn.pad_sequence(labels.split(tuple(counts.tolist()), dim=0), batch_first=True)[0::3, :, :]
        #
        # ll = torch.nn.utils.rnn.pad_sequence(orig_locs[:, 1:].split(tuple(counts.tolist()), dim=0), batch_first=True)
        # l1 = ll[0::3, :, :]
        # valid_comparisons = (l1[:, :, 2].sub(l1[:, :, 0]) * l1[:, :, 3].sub(l1[:, :, 1]) > 1024)
        # all_valid_comparisons = (orig_locs[:, 3].sub(orig_locs[:, 1]) * orig_locs[:, 4].sub(orig_locs[:, 2]) > 1024)
        # ol = torch.nn.utils.rnn.pad_sequence(meta['ori_boxes'][:, 1:].split(tuple(counts.tolist()), dim=0), batch_first=True)
        # o1 = torch.cat([ol[0::3, :, 0::2].mean(-1,True),ol[0::3, :, 1::2].mean(-1,True)],dim=-1)
        # o2 = torch.cat([ol[1::3, :, 0::2].mean(-1,True),ol[1::3, :, 1::2].mean(-1,True)],dim=-1)
        # weight = (-torch.cdist(o1,o2,p=2.0)).exp()
        #
        # #### best match over time (inter bag)
        # # inter_bag_best_match = (0.95-(torch.einsum("ijk,ikl->ijl",i1,i2.permute(0,2,1)).max(-1)[0] * valid_comparisons).max(-1)[0]).clamp(min=0.0).mean()
        # inter_bag_best_match = ((torch.einsum("ijk,ikl->ijl", i1, i2.permute(0, 2, 1)) * weight).max(-1)[0][valid_comparisons])
        #
        # #### Negative match with all samples in negative batch (inter bag)
        # # inter_bag_neg_match = torch.einsum("ijk,ikl->ijl", i1, i3.permute(0, 2, 1))[valid_comparisons]
        # # inter_bag_neg_match = (inter_bag_neg_match - 0.5).clamp(min=0.0).sum()/((inter_bag_neg_match != 0).sum()+0.000001)
        # inter_bag_neg_match = torch.einsum("ijk,ikl->ijl", i1, i3.permute(0, 2, 1))
        # inter_bag_neg_match[inter_bag_neg_match==0] = 1.0
        # inter_bag_neg_match = (inter_bag_neg_match.min(-1)[0][valid_comparisons])
        #
        # #### minor difference between samples within the bag (intra bag)
        # # intra_bag_neg_match = torch.einsum("ijk,ikl->ijl", i1, i1.permute(0, 2, 1)).triu(diagonal=1)[valid_comparisons]
        # # intra_bag_neg_match = (intra_bag_neg_match -0.9).clamp(min=0.0).sum() / ((intra_bag_neg_match != 0).sum() + 0.000001)
        #
        # p1 = bag_pred[0::3, :]
        # b1 = bag_label[0::3, :]
        # # u1 = uncertainity_score[0::3,:]
        #
        #
        # # loss = (torch.exp(-u1) * loss_fun(p1, b1) + u1).mean()
        # loss_weight = torch.tensor([1.0,0.8,0.6,0.4,0.2])[min(cur_epoch,4)]
        # # loss = (torch.exp(uncertainity_score) * loss_fun(bag_pred, bag_label) + uncertainity_score).mean()
        # # loss = loss_fun(p1, b1)
        ## contrastive loss

        #
        #
        # ## neg_loss
        # # neg_los = (loss_soft_plus(neg_uncertainity[0::3,:]).exp() * loss_fun(neg_instant[0::3,:], torch.zeros_like(neg_instant[0::3,:],requires_grad=False)) - loss_soft_plus(neg_uncertainity[0::3,:])).mean()
        # neg_los = loss_fun(neg_instant[0::3,:], torch.zeros_like(neg_instant[0::3,:],requires_grad=False)).mean()
        #
        # loss += neg_los
        #
        # # check Nan Loss.
        # # misc.check_nan_losses(inter_bag_best_match)
        # # misc.check_nan_losses(inter_bag_neg_match)
        # # misc.check_nan_losses(intra_bag_neg_match)
        # misc.check_nan_losses(neg_los)
        # misc.check_nan_losses(inst_contrast_loss)
        misc.check_nan_losses(contrast_loss)
        misc.check_nan_losses(sim_loss)
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        # Update the parameters.
        optimizer.step()
        del bag_feat, bag_label, bag_pred, contrast_loss, counts, counts_box, feat, input_cpu, inputs, inter_bag_best_match
        del inter_bag_neg_match, l1, l2, labels, meta, meta_boxes, meta_boxes_ori, neg_bag_feat, neg_counts, neg_feat
        del neg_feat_normalize, neg_inputs, neg_labels, neg_meta_boxes, neg_preds, o1, o2, orig_feat_normalize, orig_inputs
        del orig_valid, pos_bag_feat, pos_counts, pos_feat, pos_feat_normalize, pos_inputs, pos_labels, pos_meta_boxes
        del pos_meta_boxes_ori, pos_preds, preds, sim_loss, weight
        #### Get the dictionary
        if dictionary_queue is not None:
            with torch.no_grad():
                model.eval()
                if cur_epoch >= 1 and (cur_iter + 1) % 4 == 0:
                    d_ = []
                    for i, d in enumerate(zip(dictionary_queue.queue_frames_fast[:dictionary_queue.queue_samples].clone().split(16, dim=0),
                                              dictionary_queue.queue_locs[:dictionary_queue.queue_samples].clone().split(16, dim=0))):
                        d[-1][:, :, 0] -= i * 16
                        d_valid = d[-1][:, :, 1:].sum(-1) > 0
                        fr_indx = torch.linspace(0, d[0].shape[2] - 1, d[0].shape[2] // cfg.SLOWFAST.ALPHA).long()
                        img_fast = d[0].cuda().float()
                        img_slow = img_fast[:, :, fr_indx, :, :]
                        d_.append(model([img_slow, img_fast], d[-1][d_valid].float()))
                    d1, d2 = zip(*d_)
                    dict_preds, dict_feat = torch.cat(d1, dim=0), torch.cat(d2, dim=0)
                    del d1, d2, img_fast, img_slow, d, d_
            model.train()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds,_ = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds * meta["scores"]
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED + int(time.time()))
    torch.manual_seed(cfg.RNG_SEED + int(time.time()))

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    # model = moco_wrapper.MoCo(build_model(cfg), K=256)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)
        # misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    ### Add moco wrapper
    # # model = moco_wrapper.MoCo(model, K=256)
    # dict_model = Queue_data(cfg=cfg)
    # if cfg.NUM_GPUS:
    #     # Determine the GPU used by the current process
    #     cur_device = torch.cuda.current_device()
    #     # Transfer the model to the current GPU device
    # dict_model = dict_model.cuda(device=cur_device)
    dict_model = None
    torch.backends.cudnn.benchmark = True
    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )


        # Shuffle te dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        try:
            epoch_timer.epoch_tic()
            train_epoch(
                train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer, dictionary_queue=dict_model
            )
            epoch_timer.epoch_toc()
        except :
            pass
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
