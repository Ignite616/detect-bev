# -*- coding: utf-8 -*-
import torch
import numpy as np
import os

from mmdet.datasets import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .dataset_wrappers import MultiViewMixin
from IPython import embed
import mmcv
import skimage.io
import matplotlib.pyplot as plt
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import cv2
import imageio
from PIL import Image
import ipdb


def tofloat(x):
    return x.astype(np.float32) if x is not None else None


@DATASETS.register_module()
class NuScenesMultiViewDataset(MultiViewMixin, NuScenesDataset):
    def get_data_info(self, index):
        data_info = super().get_data_info(index)
        n_cameras = len(data_info['img_filename'])  # 4x6=24
        if not self.sequential:
            assert n_cameras == 6

        new_info = dict(
            sample_idx=data_info['sample_idx'],
            img_prefix=[None] * n_cameras,
            img_info=[dict(filename=x) for x in data_info['img_filename']],
            lidar2img=dict(
                extrinsic=[tofloat(x) for x in data_info['lidar2img']],
                intrinsic=np.eye(4, dtype=np.float32),
                lidar2img_aug=data_info['lidar2img_aug'],
                lidar2img_extra=data_info['lidar2img_extra']
            )
        )
        if 'ann_info' in data_info:
            gt_bboxes_3d = data_info['ann_info']['gt_bboxes_3d']
            gt_labels_3d = data_info['ann_info']['gt_labels_3d'].copy()
            mask = gt_labels_3d >= 0
            gt_bboxes_3d = gt_bboxes_3d[mask]
            gt_names = data_info['ann_info']['gt_names'][mask]
            gt_labels_3d = gt_labels_3d[mask]
            new_info['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_names=gt_names,
                gt_labels_3d=gt_labels_3d
            )
        return new_info

    def evaluate(self, results, *args, **kwargs):
        # update boxes with zero velocity
        new_results = []
        for i in range(len(results)):
            box_type = type(results[i]['boxes_3d'])
            boxes_3d = results[i]['boxes_3d'].tensor
            boxes_3d = box_type(boxes_3d, box_dim=9, origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)
    
            new_results.append(dict(
                boxes_3d=boxes_3d,
                scores_3d=results[i]['scores_3d'],
                labels_3d=results[i]['labels_3d']
            ))
        
        vis_mode = kwargs['vis_mode'] if 'vis_mode' in kwargs else False
        if vis_mode:
            embed(header='### vis nus test data ###')
            print('### vis nus test data ###')
            self.show(new_results, 'trash/test', thr=0.3)
            print('### finish vis ###')
            exit()
            
        if 'vis_mode' in kwargs.keys():
            kwargs.pop('vis_mode')
        
        result_dict = super().evaluate(new_results, *args, **kwargs)
        return result_dict
    
    # 在图像上画2d框，并且判断该框是否在多相机的交叉区域；只在图像上画交叉区域的框
    @staticmethod
    def draw_corners(img, corners, color, projection):
        corners_3d_4 = np.concatenate((corners, np.ones((8, 1))), axis=1) # (8,4)
        corners_2d_3 = corners_3d_4 @ projection.T # 这里为什么要右乘，而不是左乘？ # （8,3）
        z_mask = corners_2d_3[:, 2] > 0 # z_mask为图像前方的点 # (8,)
        corners_2d = corners_2d_3[:, :2] / corners_2d_3[:, 2:] # ()
        corners_2d = corners_2d.astype(np.int) # 转为整型  
        cross_num = 0
        for i in range(0, 8):
            if z_mask[i]==False:
                cross_num = cross_num + 1
        # 当该标注框在1-7个点不在当前图片上时
        # if cross_num >0 and cross_num < 8:
        for i, j in [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
            ]:
            # 画线时应该是and，但在画交叉区域时，一条线的两个端点难以同时在一张图片上
            if z_mask[i] and z_mask[j]: 
                img = cv2.line(
                    img=img,
                    pt1=tuple(corners_2d[i]),
                    pt2=tuple(corners_2d[j]),
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)
        # drax `X' in the front
        if z_mask[0] and z_mask[5]:
            img = cv2.line(
                img=img,
                pt1=tuple(corners_2d[0]),
                pt2=tuple(corners_2d[5]),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)
        if z_mask[1] and z_mask[4]:
            img = cv2.line(
                img=img,
                pt1=tuple(corners_2d[1]),
                pt2=tuple(corners_2d[4]),
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)
        if cross_num >0 and cross_num < 8:        
            cross_croners = corners
            return cross_croners
        else:
            return None

    def draw_bev_bbox_corner(self, img, box, color, scale_fac):
        box = box[:, None, :]  # [4,1,2]
        box = box + 50
        box = box * scale_fac
        box = np.int0(box)
        img = cv2.polylines(img, [box], isClosed=True, color=color, thickness=2)
        return img
    
    # 在bev视角，用数字表示预测类别不是car的物体
    def draw_bev_class(self, img, box, score, label, color, scale_fac, thickness=2):
        box = box[:, None, :]  # [4,1,2]
        box = box + 50
        box = box * scale_fac
        box_center_x = np.int0(box[2][0][0])
        box_center_y = np.int0((box[0][0][1]+box[2][0][1])/2)
        box = np.int0(box)
        img = cv2.polylines(img, [box], isClosed=True, color=color, thickness=2)
        score_two = round(score,2) 
        score_two = str(score_two)
        score_list = score_two.split(".")
        # 如果是1.00则画1,否则画小数点前两位
        if score_list[0] =='1':
            draw_score = score_list[0]
        else:
            draw_score = score_list[1] 
        draw_label = str(label)
        draw_score = draw_score + '_' + draw_label
        cv2.putText(img,draw_score,(box_center_x,box_center_y),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,255),thickness=1)
        return img
    
    # 在bev视角，box的中心，标出置信分数
    def draw_bev_bbox_score(self, img, box, score, color, scale_fac, thickness=2):
        box = box[:, None, :]  # [4,1,2]
        box = box + 50
        box = box * scale_fac
        box_center_x = np.int0(box[2][0][0])
        box_center_y = np.int0((box[0][0][1]+box[2][0][1])/2)
        box = np.int0(box)
        img = cv2.polylines(img, [box], isClosed=True, color=color, thickness=2)
        score_two = round(score,2) 
        # print('draw_score')
        # print(score_two)
        score_two = str(score_two)
        score_list = score_two.split(".")
        # 如果是1.00则画1,否则画小数点前两位
        if score_list[0] =='1':
            draw_score = score_list[0]
        else:
            draw_score = score_list[1] 
        # print(draw_score)
        # puttext(img, score, (box_center_x,box_center_y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),thickness=1)
        cv2.putText(img,draw_score,(box_center_x,box_center_y),cv2.FONT_HERSHEY_DUPLEX,0.6,(0,0,255),thickness=1)
        return img

    def show(self, results, out_dir='trash', bev_seg_results=None, thr=0.3, fps=3):
        assert out_dir is not None, 'Expect out_dir, got none.'
        colors = get_colors()
        all_img_gt, all_img_pred, all_bev_gt, all_bev_pred = [], [], [], []
        all_bev_all=[]
        all2_bev_all = []
        all_img_front,all_img_back = [],[]
        for i, result in enumerate(results):
            info = self.get_data_info(i)
            gt_bboxes = self.get_ann_info(i)
            print('saving image {}/{} to {}'.format(i, len(results), out_dir))
            # draw 3d box in BEV
            scale_fac = 10
            out_file_dir = str(i)
            ###### draw BEV pred ######
            bev_pred_img = np.zeros((100*scale_fac, 100*scale_fac, 3))
            if bev_seg_results is not None:
                bev_pred_road, bev_pred_lane = bev_seg_results[i]['seg_pred_road'], bev_seg_results[i]['seg_pred_lane']
                bev_pred_img = map2lssmap(bev_pred_road, bev_pred_lane)
                bev_pred_img = mmcv.imresize(bev_pred_img,
                                             (100*scale_fac, 100*scale_fac),
                                             interpolation='bilinear')
                
            scores = result['scores_3d'].numpy()
            try:
                bev_box_pred = result['boxes_3d'].corners.numpy()[:, [0, 2, 6, 4]][..., :2][scores > thr]
                labels = result['labels_3d'].numpy()[scores > thr]
                pred_bev_color = (0,255,0) # BGR
                assert bev_box_pred.shape[0] == labels.shape[0]
                for idx in range(len(labels)):
                    # bev_pred_img = self.draw_bev_bbox_corner(bev_pred_img, bev_box_pred[idx], colors[labels[idx]], scale_fac)
                    if labels[idx] == 0: 
                        bev_pred_img = self.draw_bev_bbox_score(bev_pred_img, bev_box_pred[idx], scores[idx], pred_bev_color, scale_fac)
                    else:
                        bev_pred_img = self.draw_bev_class(bev_pred_img, bev_box_pred[idx], scores[idx], labels[idx], pred_bev_color, scale_fac)
                        # bev_pred_img = self.draw_bev_bbox_corner(bev_pred_img, bev_box_pred[idx], pred_bev_color, scale_fac) # 官方画bev角点的代码
            except:
                pass
            bev_pred_for_all = bev_pred_img
            # bev_pred_img = process_bev_res_in_front(bev_pred_img)
            imsave(os.path.join(out_dir, out_file_dir, 'bev_pred.png'), bev_pred_img)

            bev_gt_img = np.zeros((100*scale_fac, 100*scale_fac, 3))
            if bev_seg_results is not None:
                sample_token = self.get_data_info(i)['sample_idx']
                bev_seg_gt = self._get_map_by_sample_token(sample_token).astype('uint8')
                bev_gt_road, bev_gt_lane = bev_seg_gt[...,0], bev_seg_gt[...,1]
                bev_seg_gt = map2lssmap(bev_gt_road, bev_gt_lane)
                bev_gt_img = mmcv.imresize(
                    bev_seg_gt,
                    (100*scale_fac, 100*scale_fac),
                    interpolation='bilinear')
            try:
                # draw BEV GT
                bev_gt_bboxes = gt_bboxes['gt_bboxes_3d'].corners.numpy()[:,[0,2,6,4]][..., :2]
                labels_gt = gt_bboxes['gt_labels_3d']
                gt_bev_color = (0,128,255) # BGR
                for idx in range(len(labels_gt)):
                    # bev_gt_img = self.draw_bev_bbox_corner(bev_gt_img, bev_gt_bboxes[idx], colors[labels_gt[idx]], scale_fac)
                    bev_gt_img = self.draw_bev_bbox_corner(bev_gt_img, bev_gt_bboxes[idx], gt_bev_color, scale_fac)
            except:
                pass
            bev_gt_img = process_bev_res_in_front(bev_gt_img)
            imsave(os.path.join(out_dir, out_file_dir, 'bev_gt.png'), mmcv.imrescale(bev_gt_img, 0.5))
            all_bev_gt.append(mmcv.imrescale(bev_gt_img, 0.5))
            all_bev_pred.append(mmcv.imrescale(bev_pred_img, 0.5))
            # 将bev_pre 和 bev_gt画在一张图上,在bev_pre的图上画bev_gt即可
            try:
                bev_gt_bboxes = gt_bboxes['gt_bboxes_3d'].corners.numpy()[:,[0,2,6,4]][..., :2]
                labels_gt = gt_bboxes['gt_labels_3d']
                for idx in range(len(labels_gt)):
                    bev_all_img = self.draw_bev_bbox_corner(bev_pred_for_all, bev_gt_bboxes[idx], gt_bev_color, scale_fac)
            except:
                pass
            # bev_all_img = process_bev_res_in_front(bev_all_img)
            imsave(os.path.join(out_dir, out_file_dir, 'bev_all.png'), bev_all_img)
            all_bev_all.append(mmcv.imrescale(bev_all_img, 0.5))
            ###### draw 3d box in image ######
            img_gt_list = []
            img_pred_list = []
            cross_pred_list = []
            cross_gt_list = []
            img_origin_list = [] # 存放原始的六张图片
            for j in range(min(len(info['img_info']),6)):
                img_pred = imread(info['img_info'][j]['filename'])
                img_gt = imread(info['img_info'][j]['filename'])
                # camera name
                # print(info['img_info'][j]['filename'])
                camera_name = info['img_info'][j]['filename'].split('/')[-2]
                puttext(img_pred, camera_name)
                puttext(img_gt, camera_name)
                # 保存原图,img_gt之后没有被重写
                img_origin_list.append(mmcv.imrescale(img_gt, 0.5))
                # imsave(os.path.join(out_dir, out_file_dir, '{}_origin.png'.format(j)), mmcv.imrescale(img_pred, 0.5))
                extrinsic = info['lidar2img']['extrinsic'][j]
                intrinsic = info['lidar2img']['intrinsic'][:3, :3]
                projection = intrinsic @ extrinsic[:3]
                if not len(result['scores_3d']):
                    pass
                else:
                    # draw pred 3d box in image
                    corners = result['boxes_3d'].corners.numpy()
                    scores = result['scores_3d'].numpy()
                    labels = result['labels_3d'].numpy()
                    for corner, score, label in zip(corners, scores, labels):
                        if score < thr:
                            continue
                        try:
                            # self.draw_corners(img_pred, corner, colors[label], projection)
                            cross_pred = self.draw_corners(img_pred, corner, pred_bev_color, projection)
                            if cross_pred is not None:
                                cross_pred_list.append(cross_pred)
                        except:
                            pass
                    img_pred_for_gt = img_pred
                    imsave(os.path.join(out_dir, out_file_dir, '{}_pred.png'.format(j)), mmcv.imrescale(img_pred, 0.5))
                    img_pred_list.append(mmcv.imrescale(img_pred, 0.5))
                    # draw GT 3d box in image
                    corners = gt_bboxes['gt_bboxes_3d'].corners.numpy()
                    labels = gt_bboxes['gt_labels_3d']
                    for corner, label in zip(corners, labels):
                        try:
                            # self.draw_corners(img_gt, corner, colors[label], projection)
                            cross_gt = self.draw_corners(img_pred_for_gt, corner, gt_bev_color, projection) # 和pred的结果画到一张图片上
                            if cross_gt is not None:
                                cross_gt_list.append(cross_gt)
                        except:
                            pass
                out_file_dir = str(i)
                mmcv.mkdir_or_exist(os.path.join(out_dir, out_file_dir))
                # 缩小image大小 可视化方便一些;将真值box,和预测的box拼接起来.预测的在下方
                # img_gt_pred = np.concatenate([img_gt, img_pred], 0)
                # imsave(os.path.join(out_dir, out_file_dir, '{}_gt_pred.png'.format(j)), mmcv.imrescale(img_gt_pred, 0.5))
                imsave(os.path.join(out_dir, out_file_dir, '{}_gt_pred.png'.format(j)), mmcv.imrescale(img_pred_for_gt, 0.5))
                # img_gt_list.append(mmcv.imrescale(img_gt, 0.5))
            ###### draw 3d box in image ######

            # 将bev_pre 和 bev_gt中的交叉区域画在一张图上,在bev_pre的图上画bev_gt即可;先画pred，再画gt
            bev_pred_img2 = np.zeros((100*scale_fac, 100*scale_fac, 3))
            scores = result['scores_3d'].numpy()
            if cross_pred_list:
                cross_pred_numpy = np.array(cross_pred_list)
                bev_box_pred2 = cross_pred_numpy[:, [0, 2, 6, 4], 0:2]
                for idx in range(len(cross_pred_list)):
                # bev_pred_img = self.draw_bev_bbox_corner(bev_pred_img, bev_box_pred[idx], colors[labels[idx]], scale_fac)
                    bev_pred_for_all2 = self.draw_bev_bbox_score(bev_pred_img2, bev_box_pred2[idx], scores[idx], pred_bev_color, scale_fac)
            else:
                bev_pred_for_all2 = bev_pred_img2
            # 继续画交叉区域的bev_gt
            if cross_gt_list:
                cross_gt_numpy = np.array(cross_gt_list)
                bev_gt_bboxes2 = cross_gt_numpy[:,[0,2,6,4], 0:2]
                try:
                    for idx in range(len(cross_gt_list)):
                        bev_all2_img = self.draw_bev_bbox_corner(bev_pred_for_all2, bev_gt_bboxes2[idx], gt_bev_color, scale_fac)
                except:
                    pass
            else:
                bev_all2_img = bev_pred_for_all2
            
            bev_all2_img = process_bev_res_in_front(bev_all2_img)
            imsave(os.path.join(out_dir, out_file_dir, 'bev_all2.png'), mmcv.imrescale(bev_all2_img, 0.5))
            all2_bev_all.append(mmcv.imrescale(bev_all2_img, 0.5))

            ###### generate videos step:1 ######  上面三张是原始图片，下面三张是预测结果；
            # 保存当前帧的前视角,后视角的3张图,由于视频可视化; sort_list是为了排序,实现左中右的效果
            # 前后拼接是为了实现,同一帧的,前视角三张,后视角三张的拼接.
            # 前三张的原图和预测
            tmp_img_front_pred = np.concatenate(sort_list(img_pred_list[0:3], sort=[2,0,1]), axis=1)  
            tmp_img_front_origin = np.concatenate(sort_list(img_origin_list[0:3], sort=[2,0,1]), axis=1)  
            tmp_img_front = np.concatenate([tmp_img_front_origin, tmp_img_front_pred], axis=0)
            all_img_front.append(tmp_img_front)
            # 后三张的原图和预测
            tmp_img_back_pred = np.concatenate(sort_list(img_pred_list[3:6], sort=[2,0,1]) ,axis=1) #TODO:修改为倒数三个
            tmp_img_back_origin = np.concatenate(sort_list(img_origin_list[3:6], sort=[2,0,1]), axis=1) 
            tmp_img_back = np.concatenate([tmp_img_back_origin, tmp_img_back_pred], axis=0)
            all_img_back.append(tmp_img_back)
            # all_img_pred.append(tmp_img_pred)
            # tmp_img_up_gt = np.concatenate(sort_list(img_gt_list[0:3], sort=[2,0,1]),axis=1)
            # # tmp_img_bottom_gt = np.concatenate(sort_list(img_gt_list[-3:], sort=[2,0,1]),axis=1)
            # tmp_img_bottom_gt = np.concatenate(sort_list(img_gt_list[3:6], sort=[2,0,1]),axis=1) #TODO:修改为倒数三个
            # tmp_img_gt = np.concatenate([tmp_img_up_gt, tmp_img_bottom_gt], axis=0)
            # all_img_gt.append(tmp_img_gt)
            ###### generate videos step:1 ######
        ###### generate videos step:2 ######
        # gen_video(all_img_pred, all_bev_pred, out_dir, 'pred', fps=fps)
        # gen_video(all_img_gt, all_bev_gt, out_dir, 'gt', fps=fps)
        # gen_video(all_img_pred, all_bev_all, out_dir, 'all', fps=fps)
        # gen_video(all_img_pred, all2_bev_all, out_dir, 'all2', fps=fps)
        ### 同时保存预测值，和原照片
        gen_video(all_img_front, all_bev_all, out_dir, 'all', fps=fps)  # 前面的三张图
        gen_video(all_img_back, all_bev_all, out_dir, 'all2', fps=fps)  # 后面的三张图
        ###### generate videos step:2 ######

def sort_list(_list, sort):
    assert len(_list) == len(sort)
    new_list = []
    for s in sort:
        new_list.append(_list[s])
    return new_list


def get_colors():
    colors = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8).tolist()
    colors = [i[::-1] for i in colors]
    return colors


def imread(img_path):
    img = mmcv.imread(img_path)
    return img


def imsave(img_path, img):
    mmcv.imwrite(img, img_path, auto_mkdir=True)


def puttext(img, name, loc=(30, 60), font=cv2.FONT_HERSHEY_DUPLEX ,color=(248, 202, 105)):
    try:
        cv2.putText(img, name, loc, font, 2, color, 2)
    except:
        img = Image.fromarray(img)
        img = np.array(img)
        cv2.putText(img, name, loc, font, 2, color, 2)

        
def map2lssmap(bev_map_road, bev_map_lane):
    white = [200, 200, 200]
    orange = [80, 127, 255]
    green = [171, 193, 115]
    bev_map = np.zeros_like(bev_map_road)[:,:,None].repeat(3,-1).astype('uint8')
    bev_map[bev_map[:, :, 0] == 0] = white
    bev_map[bev_map_road == 1] = orange
    bev_map[bev_map_lane == 1] = green
    return bev_map


def gen_video(img_list, bev_list, out_dir, mode='pred', fps=3):
    if mode == 'pred':
        out_video_path = os.path.join(out_dir, 'video_pred.mp4')
    elif mode == 'gt':
        out_video_path = os.path.join(out_dir, 'video_gt.mp4')
    elif mode == 'all2':
        out_video_path = os.path.join(out_dir, 'video_all2.mp4')
    else:
        assert mode == 'all'
        out_video_path = os.path.join(out_dir, 'video_all.mp4')
    tmp_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        bev = bev_list[i]
        bev = draw_ego_car(bev)
        bev = mmcv.imrescale(bev, img.shape[0]/bev.shape[0])
        # img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        # bev = cv2.copyMakeBorder(bev, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        tmp = np.concatenate([img, bev], axis=1)
        tmp = tmp[:, :, ::-1].astype('uint8')
        tmp_list.append(tmp)
        
    imageio.mimsave(out_video_path, tmp_list, fps=fps)
    print('finish video generation')
    print('video path: {}'.format(out_video_path))


def process_bev_res_in_front(bev):
    bev = np.flip(bev, axis=0)
    return bev


def draw_ego_car(bev):
    h, w, _ = bev.shape
    ego_h = 8
    ego_w = 20
    x1 = (h - ego_h) // 2
    y1 = (w - ego_w) // 2
    x2 = (h + ego_h) // 2
    y2 = (w + ego_w) // 2
    color = [255, 0, 0]
    bev = cv2.rectangle(bev, (x1, y1), (x2, y2), color, -1)
    return bev
