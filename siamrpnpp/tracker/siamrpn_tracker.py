# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
from PIL import Image
from toolkit.utils.statistics import iou


from siamrpnpp.tracker.classifier.libs.plotting import show_tensor
from siamrpnpp.tracker.classifier.base_classifier import BaseClassifier

from siamrpnpp.core.config import cfg
from siamrpnpp.utils.anchor import Anchors
from siamrpnpp.tracker.base_tracker import SiameseTracker


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.lost_count = 0

        self.model = model
        self.model.eval()

        if cfg.TRACK.USE_CLASSIFIER:                        
            self.classifier = BaseClassifier(self.model)


    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height


    def init(self, img, bbox):
        self.frame_num = 1
        self.temp_max = 0

        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        self.channel_average = np.mean(img, axis=(0, 1))
        self.z0_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE,
            s_z, self.channel_average)
        self.z_crop = self.z0_crop

        with torch.no_grad():
            self.model.template(self.z0_crop)

        if cfg.TRACK.USE_CLASSIFIER:                        #模板更新部分
            if cfg.TRACK.TEMPLATE_UPDATE:
                with torch.no_grad():
                    self.model.template_short_term(self.z_crop)

            s_xx = s_z * (cfg.TRACK.INSTANCE_SIZE * 2 / cfg.TRACK.EXEMPLAR_SIZE)
            x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE * 2,
                round(s_xx), self.channel_average)

            self.classifier.initialize(x_crop.type(torch.FloatTensor), bbox)



    def track(self, img):
        self.frame_num += 1
        self.curr_frame = img
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
                round(s_x), self.channel_average)

        with torch.no_grad():
            outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        def normalize(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
 
        #1
        if cfg.TRACK.USE_CLASSIFIER:
            flag, s = self.classifier.track()
            if flag == 'not_found':
                self.lost_count += 1
            else:
                self.lost_count = 0

            confidence = Image.fromarray(s.detach().cpu().numpy())
            confidence = np.array(confidence.resize((self.score_size, self.score_size))).flatten()
            pscore = pscore.reshape(5, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                normalize(confidence) * cfg.TRACK.COEE_CLASS
            pscore = pscore.flatten()

            if cfg.TRACK.TEMPLATE_UPDATE:
                score_st = self._convert_score(outputs['cls_st'])
                pred_bbox_st = self._convert_bbox(outputs['loc_st'], self.anchors)
                s_c_st = change(sz(pred_bbox_st[2, :], pred_bbox_st[3, :]) /
                                (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
                r_c_st = change((self.size[0] / self.size[1]) /
                                (pred_bbox_st[2, :] / pred_bbox_st[3, :]))
                penalty_st = np.exp(-(r_c_st * s_c_st - 1) * cfg.TRACK.PENALTY_K)
                pscore_st = penalty_st * score_st
                pscore_st = pscore_st.reshape(5, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                            normalize(confidence) * cfg.TRACK.COEE_CLASS
                pscore_st = pscore_st.flatten()
                
        #window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        #2
        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
            cx, cy = bbox[0] / 4 + self.center_pos[0], bbox[1] / 4 + self.center_pos[1]
        else:
            cx, cy = bbox[0] + self.center_pos[0], bbox[1] + self.center_pos[1]
        
        #smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        #3
        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.TEMPLATE_UPDATE:
            pscore_st = pscore_st * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx_st = np.argmax(pscore_st)
            bbox_st = pred_bbox_st[:, best_idx_st] / scale_z
            lr_st = penalty_st[best_idx_st] * score_st[best_idx_st] * cfg.TRACK.LR
            if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
                cx_st, cy_st = bbox_st[0] / 4 + self.center_pos[0], bbox_st[1] / 4 + self.center_pos[1]
            else:
                cx_st, cy_st = bbox_st[0] + self.center_pos[0], bbox_st[1] + self.center_pos[1]
            width_st = self.size[0] * (1 - lr_st) + bbox_st[2] * lr_st
            height_st = self.size[1] * (1 - lr_st) + bbox_st[3] * lr_st
            cx_st, cy_st, width_st, height_st = self._bbox_clip(cx_st, cy_st, width_st, height_st, img.shape[:2])
            if iou((cx_st, cy_st, width_st, height_st), (cx, cy, width, height), wh=True) >= cfg.TRACK.TAU_REGRESSION \
                and score_st[best_idx_st] - score[best_idx] >= cfg.TRACK.TAU_CLASSIFICATION:
                cx, cy, width, height, score, best_idx = cx_st, cy_st, width_st, height_st, score_st, best_idx_st


        #udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2, cy - height / 2,
                width, height]
        best_score = score[best_idx]

        #4
        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier.update(bbox, scale_z, flag)

            if cfg.TRACK.TEMPLATE_UPDATE:
                if torch.max(s).item() >= cfg.TRACK.TARGET_UPDATE_THRESHOLD and flag != 'hard_negative':
                    if torch.max(s).item() > self.temp_max:
                        self.temp_max = torch.max(s).item()
                        self.channel_average = np.mean(img, axis=(0, 1))
                        self.z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)

                if (self.frame_num - 1) % cfg.TRACK.TARGET_UPDATE_SKIPPING == 0:
                    self.temp_max = 0
                    with torch.no_grad():
                        self.model.template_short_term(self.z_crop)

        if cfg.TRACK.USE_CLASSIFIER:
            return {
                    'bbox': bbox,
                    'best_score': best_score,
                    'flag': flag
                   }
        else:
            return {
                    'bbox': bbox,
                    'best_score': best_score
                   }