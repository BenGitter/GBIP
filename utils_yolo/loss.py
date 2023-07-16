# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_yolo.general import bbox_iou, box_iou, xywh2xyxy
from utils_yolo.torch_utils import is_parallel
from utils_gbip.adversarial import AdversarialGame

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputeLossOTA:
    # Compute losses
    def __init__(self, model, OT=True, AT=True, AG=True, model_T=None):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        if AG:
            self.adversarial_game = AdversarialGame(device)

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        KLDcls_OT = nn.KLDivLoss(reduction='batchmean', log_target=True) # KL Divergence Transfer Learning: classification
        BCEobj_OT = nn.BCEWithLogitsLoss() # Transfer Learning: objectness

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi =  0
        self.BCEcls, self.BCEobj, self.KLDcls_OT, self.BCEobj_OT, self.gr, self.hyp, self.model_T, self.OT, self.AT, self.AG = \
            BCEcls, BCEobj, KLDcls_OT, BCEobj_OT, model.gr, h, model_T, OT, AT, AG
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs, att=None):  # predictions, targets, images, attention maps
        device = targets.device
        l = torch.zeros(9, device=device)
        lcls, lbox, lobj, lbox_tl, lcls_tl, lobj_tl, lat, lag, lmg = range(9)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p] 

        # Create local AT variable; for validation att is None and so we don't run AT stuff.
        AT = self.AT and att
        # Run Teacher model with same inputs if AT and/or OT is enabled
        with torch.no_grad():
            if AT:
                assert att is not None  # make sure student attention maps are included
                pred_T, att_T = self.model_T(imgs, AT=True, attention_layers=self.hyp['attention_layers'])
            else:
                if self.OT or self.AG:
                    pred_T = self.model_T(imgs)[1]

        if AT:
            for j, map_S in enumerate(att):
                map_T = att_T[j]

                # Calculate loss for this map
                A_S = map_S.pow(2).mean(1).view(-1)
                A_T = map_T.pow(2).mean(1).view(-1)
                l[lat] += torch.norm((A_S / A_S.norm(2)) - (A_T / A_T.norm(2)), 2)

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                l[lbox] += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    l[lcls] += self.BCEcls(ps[:, 5:], t)  # BCE

            # Transfer Learning: IoU loss + KL divergence classification + BCELoss objectness 
            if self.OT:
                # create boxes for Teacher
                pflat_T = pred_T[i].view(-1, 3, 85)
                pxy_T = pflat_T[:, :, :2].sigmoid() * 2. - 0.5
                pwh_T = (pflat_T[:, :, 2:4].sigmoid() * 2) ** 2 * self.anchors[i]
                pbox_T = torch.cat((pxy_T.view(-1, 2), pwh_T.view(-1, 2)), 1)
                # create boxes for Student
                pflat_S = pi.view(-1, 3, 85)
                pxy_S = pflat_S[:, :, :2].sigmoid() * 2. - 0.5
                pwh_S = (pflat_S[:, :, 2:4].sigmoid() * 2) ** 2 * self.anchors[i]
                pbox_S = torch.cat((pxy_S.view(-1, 2), pwh_S.view(-1, 2)), 1)
                # calculate IoU loss between S and T
                iou_TL = bbox_iou(pbox_S.T, pbox_T, x1y1x2y2=False, CIoU=True)
                l[lbox_tl] += (1.0 - iou_TL).mean()

                # calculate classification loss between S and T
                kl_cls_input = F.log_softmax(pi[:,:,:,:,5:].view(-1, 80) / self.hyp['OT_temp'], dim=1)
                kl_cls_target = F.log_softmax(pred_T[i][:,:,:,:,5:].view(-1, 80) / self.hyp['OT_temp'], dim=1)
                l[lcls_tl] += self.hyp['OT_temp']**2 * self.KLDcls_OT(kl_cls_input, kl_cls_target)
                
                # calculate objectness loss between S and T
                kl_obj_input = pi[:,:,:,:,4].view(-1, 1)
                kl_obj_target = torch.sigmoid(pred_T[i][:,:,:,:,4].view(-1, 1))
                l[lobj_tl] += self.BCEobj_OT(kl_obj_input, kl_obj_target)
            

            obji = self.BCEobj(pi[..., 4], tobj)
            l[lobj] += obji * self.balance[i]  # obj loss

            if self.AG:
                bs_i = pi.shape[0]
                no = pi.shape[4]
                in_S = pi.reshape(bs_i, -1, no)[:, :, 5:]
                l[lag] += 1/3 * self.adversarial_game.get_student_loss(in_S)

        l[lbox] *= self.hyp['box']
        l[lobj] *= self.hyp['obj']
        l[lcls] *= self.hyp['cls']
        l[lbox_tl] *= self.hyp['box']
        l[lobj_tl] *= self.hyp['obj']
        l[lcls_tl] *= self.hyp['cls'] * self.hyp['lkl_cls']
        l[lag] *= self.hyp['lag']
        l[lat] *= self.hyp['lat']
        bs = tobj.shape[0]  # batch size

        loss = l[lbox] + l[lobj] + l[lcls]
        if self.OT:
            # loss += (l[lcls_tl] + l[lbox_tl] + l[lobj_tl]) * self.hyp['lot']
            loss += (l[lcls_tl] + l[lobj_tl]) * self.hyp['lot']
            # loss = self.hyp['OT_alpha'] * (l[lcls_tl] + l[lbox_tl] + l[lobj_tl]) + (1 - self.hyp['OT_alpha']) * loss
        if AT:
            loss += l[lat]
        if self.AG:
            loss += l[lag]

        return loss * bs, torch.cat((l[:lag+1], loss.unsqueeze(0), l[lmg].unsqueeze(0))).detach()

    def update_AG(self, imgs, pred_S):
        with torch.no_grad():
            pred_T = self.model_T(imgs)[1]

        loss_sum = torch.zeros(1).cuda()
        for i in range(3):
            bs_i = pred_S[i].shape[0]
            no = pred_S[i].shape[4]
            in_T = pred_T[i].reshape(bs_i, -1, no)[:, :, 5:]
            in_S = pred_S[i].detach().clone()
            in_S = in_S.reshape(bs_i, -1, no)[:, :, 5:]
            loss_sum += self.adversarial_game.update(in_S, in_T)
        
        return loss_sum / 3    

    def build_targets(self, p, targets, imgs):
        device = targets.device
        #indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        #indices, anch = self.find_4_positive(p, targets)
        #indices, anch = self.find_5_positive(p, targets)
        #indices, anch = self.find_9_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]
        
        nl = len(p)    
    
        for batch_idx in range(p[0].shape[0]):
        
            b_idx = targets[:, 0]==batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
                
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []
            
            for i, pi in enumerate(p):
                
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]                
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))
                fg_pred = pi[b, a, gj, gi]                
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])
                
                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i] #/ 8.
                #pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)
            
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        
            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
               torch.log(y/(1-y)) , gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_
        
            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
        
            this_target = this_target[matched_gt_inds]
        
            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs           

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
    