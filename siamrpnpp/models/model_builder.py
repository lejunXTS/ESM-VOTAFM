# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamrpnpp.core.config import cfg
from siamrpnpp.models.loss import select_cross_entropy_loss, weight_l1_loss
from siamrpnpp.models.backbone import get_backbone
from siamrpnpp.models.head import get_rpn_head, get_mask_head, get_refine_head
from siamrpnpp.models.neck import get_neck

from siamrpnpp.models.attention import GlobalAttentionBlock,CBAM,SELayer1



class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)
        
        self.tematt = GlobalAttentionBlock()       #me 注意力机制
        self.detatt = CBAM(256)
        self.att1 = nn.Sequential(SELayer1(256))    #注意力机制  SE block

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def avg(self, lst):
        return sum(lst) / len(lst)

    def weighted_avg(self, lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        zf1 = self.tematt(zf)    #模板注意力机制 
        zf2 = self.att1(zf)
        zf2 = zf2 + zf
        zf = zf1 + zf2

        self.zf = zf

    def template_short_term(self, z_st):           #模板更新部分
        zf_st = self.backbone(z_st)
        if cfg.MASK.MASK:
            zf_st = zf_st[-1]
        if cfg.ADJUST.ADJUST:
            zf_st = self.neck(zf_st)

        zf1 = self.tematt(zf_st)    #模板注意力机制 
        zf2 = self.att1(zf_st)
        zf2 = zf2 + zf_st
        zf_st = zf1 + zf2

        self.zf_st = zf_st

    def instance(self, x):        #模板更新部分
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        xf1 = self.detatt(xf)     #搜索注意力机制
        xf2 = self.att1(xf)
        xf2 = xf2 + xf
        xf = xf1 + xf2

        if not cfg.ADJUST.LAYER:
            if cfg.ADJUST.FUSE == 'wavg':
                cls_weight = self.rpn_head.cls_weight
                self.cf = self.weighted_avg([cf for cf in xf], cls_weight)
            elif cfg.ADJUST.FUSE == 'avg':
                self.cf = self.avg([cf for cf in xf])
            elif cfg.ADJUST.FUSE == 'con':
                self.cf = torch.cat([cf for cf in xf], dim=1)
        else:
            if isinstance(xf, list):
                self.cf = xf[cfg.ADJUST.LAYER-1]
            else:
                self.cf = xf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        xf1 = self.detatt(xf)     #搜索注意力机制
        xf2 = self.att1(xf)
        xf2 = xf2 + xf
        xf = xf1 + xf2

        if not cfg.ADJUST.LAYER:
            if cfg.ADJUST.FUSE == 'wavg':
                cls_weight = self.rpn_head.cls_weight
                self.cf = self.weighted_avg([cf for cf in xf], cls_weight)
            elif cfg.ADJUST.FUSE == 'avg':
                self.cf = self.avg([cf for cf in xf])
            elif cfg.ADJUST.FUSE == 'con':
                self.cf = torch.cat([cf for cf in xf], dim=1)
        else:
            if isinstance(xf, list):
                self.cf = xf[cfg.ADJUST.LAYER-1]
            else:
                self.cf = xf


        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        # return {
        #         'cls': cls,
        #         'loc': loc,
        #         'mask': mask if cfg.MASK.MASK else None
        #        }
    
        if cfg.TRACK.TEMPLATE_UPDATE:
            cls_st, loc_st = self.rpn_head(self.zf_st, xf)
            return {
                    'cls': cls,
                    'loc': loc,
                    'cls_st': cls_st,
                    'loc_st': loc_st,
                    'mask': mask if cfg.MASK.MASK else None
                   }
        else:
            return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
            }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)    
        xf = self.backbone(search)


        zf1 = self.tematt(zf)    #为模板分支和搜索分支分别设计的注意力机制
        xf1 = self.detatt(xf)
        zf2 = self.att1(zf)
        xf2 = self.att1(xf)
        
        zf2 = zf2 + zf
        xf2 = xf2 +xf
        zf = zf1 + zf2
        xf = xf1 + xf2


        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
