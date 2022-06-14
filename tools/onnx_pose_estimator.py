import cv2
import argparse
import os, csv, time, shutil
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from mmpose.core.post_processing import flip_back
import mmcv
import numpy as np
from mmpose.core.bbox import bbox_xyxy2xywh
from mmpose.datasets.pipelines import Compose
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps

import onnx
import onnxruntime as rt
import copy


class OnnxPoseEstimator:
    def __init__(self, pose_config, onnx_path):
        self.cfg = mmcv.Config.fromfile(pose_config)

        onnx_model = onnx.load(onnx_path)

        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        del onnx_model
        onnx_model = None
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        self.sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("loaded onnx model")
        self.onnx_input_key = net_feed_input[0]

        self.flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                           [16, 17], [18, 19]]

    def decode_heatmap(self, img_meta, output):
        c = np.zeros((1, 2), dtype=np.float32)
        s = np.zeros((1, 2), dtype=np.float32)
        c[0, :] = img_meta['center']
        s[0, :] = img_meta['scale']

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.cfg.get('unbiased_decoding', False),
            post_process=self.cfg.get('post_process', 'default'),
            kernel=self.cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.cfg.get('valid_radius_factor',
                                             0.0546875),
            use_udp=self.cfg.get('use_udp', False),
            target_type=self.cfg.get('target_type', 'GaussianHeatmap'))

        all_preds = np.zeros((1, preds.shape[1], 3), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        return all_preds[0]

    def _inference_single_pose_model_onnx(self, imgs_or_paths, bbox):
        _test_pipeline = copy.deepcopy(self.cfg.test_pipeline)

        has_bbox_xywh2cs = False
        for transform in _test_pipeline:
            if transform['type'] == 'TopDownGetBboxCenterScale':
                has_bbox_xywh2cs = True
                break
        if not has_bbox_xywh2cs:
            _test_pipeline.insert(
                0, dict(type='TopDownGetBboxCenterScale', padding=1.25))
        test_pipeline = Compose(_test_pipeline)

        data = {
            'bbox':
                bbox,
            'bbox_score':
                bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
                0,  # need to be assigned if batch_size > 1
            'dataset':
                self.cfg.data.test.type,
            'joints_3d':
                np.zeros((self.cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
                np.zeros((self.cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
                0,
            'ann_info': {
                'image_size': np.array(self.cfg.data_cfg['image_size']),
                'num_joints': self.cfg.data_cfg['num_joints'],
                'flip_pairs': self.flip_pairs
            }
        }

        if isinstance(imgs_or_paths, np.ndarray):
            data['img'] = imgs_or_paths
        else:
            data['image_file'] = imgs_or_paths
        data = test_pipeline(data)

        img_meta = data['img_metas'].data
        img = data['img'].unsqueeze(0)
        output_heatmap = self.sess.run(None, {self.onnx_input_key: img.detach().numpy()})
        output_heatmap = output_heatmap[0]
        flip_output_heatmap = np.array([output_heatmap[1]])
        output_heatmap = np.array([output_heatmap[0]])

        flip_output_heatmap = flip_back(
            flip_output_heatmap,
            self.flip_pairs,
            target_type=self.cfg.get('target_type', 'GaussianHeatmap'))
        # feature is not aligned, shift flipped heatmap for higher accuracy
        if self.cfg.get('shift_heatmap', False):
            flip_output_heatmap[:, :, :, 1:] = flip_output_heatmap[:, :, :, :-1]

        output_heatmap = (output_heatmap + flip_output_heatmap) * 0.5

        return self.decode_heatmap(img_meta, output_heatmap)

    def inference(self, imgs_or_paths, dog_bbox, bbox_thr=0.01):
        '''

        Args:
            imgs_or_paths: image array or image file path
            dog_bbox: dog bbox. format: [x1, y1, x2, y2, score]
            bbox_thr: bbox threshold

        Returns:
            pose result. shape: [20, 3]

        '''
        bboxes = np.array([dog_bbox])
        if bboxes[0][4] < bbox_thr:
            return None

        bboxes_xywh = bbox_xyxy2xywh(bboxes)

        if len(bboxes_xywh) == 0:
            return None

        return self._inference_single_pose_model_onnx(imgs_or_paths, bboxes_xywh[0])


def main(args):
    # saved csv only folder path of inference images (same data in args.result_dir)
    result_csv_folder = args.result_dir + '_csv_only'
    onnx_result_csv_folder = args.result_dir + '_csv_only_onnx'

    gpu_device = 'cuda'

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=gpu_device)
    # initialize detector
    det_model = init_detector(args.det_config, args.det_checkpoint, device=gpu_device)

    pose_estimator = OnnxPoseEstimator(pose_config=args.pose_config, onnx_path=args.onnx_path)

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(result_csv_folder, exist_ok=True)
    os.makedirs(onnx_result_csv_folder, exist_ok=True)

    scorer_index = ['scorer'] + (['teamDLC'] * 60)

    bodyparts_index = ['bodyparts', 'L_Eye', 'L_Eye', 'L_Eye', 'R_Eye', 'R_Eye', 'R_Eye',
                       'L_EarBase', 'L_EarBase', 'L_EarBase', 'R_EarBase', 'R_EarBase', 'R_EarBase',
                       'Nose', 'Nose', 'Nose', 'Throat', 'Throat', 'Throat', 'TailBase', 'TailBase',
                       'TailBase', 'Withers', 'Withers', 'Withers', 'L_F_Elbow', 'L_F_Elbow', 'L_F_Elbow',
                       'R_F_Elbow', 'R_F_Elbow', 'R_F_Elbow', 'L_B_Elbow', 'L_B_Elbow', 'L_B_Elbow', 'R_B_Elbow',
                       'R_B_Elbow', 'R_B_Elbow', 'L_F_Knee', 'L_F_Knee', 'L_F_Knee', 'R_F_Knee', 'R_F_Knee', 'R_F_Knee',
                       'L_B_Knee', 'L_B_Knee', 'L_B_Knee', 'R_B_Knee', 'R_B_Knee', 'R_B_Knee', 'L_F_Paw', 'L_F_Paw',
                       'L_F_Paw',
                       'R_F_Paw', 'R_F_Paw', 'R_F_Paw', 'L_B_Paw', 'L_B_Paw', 'L_B_Paw', 'R_B_Paw', 'R_B_Paw',
                       'R_B_Paw']
    coords_index = ['coords'] + (['x', 'y', 'score'] * 20)

    bodyparts_index2 = ['bodyparts', 'bbox']
    coords_index2 = ['coords', 'x1, y1, x2, y2, thr']

    time_od = []
    time_pe = []

    for folder in os.listdir(args.image_root):
        if folder[0] == '.' or folder[-1] == 's' or folder[-4:] == '.zip':  # skip strange folder
            continue

        os.makedirs(args.result_dir + '/' + folder, exist_ok=True)
        os.makedirs(result_csv_folder + '/' + folder, exist_ok=True)
        os.makedirs(onnx_result_csv_folder + '/' + folder, exist_ok=True)

        for folder2 in os.listdir(args.image_root + '/' + folder):
            if folder2[0] == '.' or folder2[-1] == 's' or folder[-4:] == '.zip':  # skip strange folder
                continue

            print('==', folder, "/", folder2)
            os.makedirs(args.result_dir + '/' + folder + '/' + folder2, exist_ok=True)
            os.makedirs(result_csv_folder + '/' + folder + '/' + folder2, exist_ok=True)
            os.makedirs(onnx_result_csv_folder + '/' + folder + '/' + folder2, exist_ok=True)

            with open(args.result_dir + '/' + folder + '/' + folder2 + '/vis_' + folder2 + '.csv', 'w',
                      newline='') as w, \
                    open(args.result_dir + '/' + folder + '/' + folder2 + '/bbox_' + folder2 + '.csv', 'w',
                         newline='') as w2, \
                    open(args.result_dir + '/' + folder + '/' + folder2 + '/vis_onnx_' + folder2 + '.csv', 'w',
                         newline='') as onnx_w:
                wr = csv.writer(w)
                wr.writerow(scorer_index)
                wr.writerow(bodyparts_index)
                wr.writerow(coords_index)

                wr2 = csv.writer(w2)
                wr2.writerow(bodyparts_index2)
                wr2.writerow(coords_index2)

                onnx_wr = csv.writer(onnx_w)
                onnx_wr.writerow(scorer_index)
                onnx_wr.writerow(bodyparts_index)
                onnx_wr.writerow(coords_index)

                img_list = os.listdir(args.image_root + '/' + folder + '/' + folder2)
                img_list.sort()

                mmdet_results_list = []
                for image in img_list:
                    if image[-1] != 'g':  # skip files with out jpg, png
                        continue

                    img = args.image_root + '/' + folder + '/' + folder2 + '/' + image

                    # test a single image, the resulting box is (x1, y1, x2, y2)
                    start = time.time()
                    mmdet_results = inference_detector(det_model, img)
                    #                 print("mmdet_results: ",mmdet_results)
                    time_od.append(time.time() - start)
                    #                 print('-OD',image, ':', time_od[-1])

                    # keep the dog class bounding boxes.
                    dog_results = process_mmdet_results(mmdet_results, cat_id=17)
                    print("dog_results", len(dog_results))
                    #                 print("dog_results: ",dog_results)

                    # save bbox results
                    row2 = [image]
                    if len(dog_results) > 0:
                        mmdet_results_list.append([dog_results[0]])
                        row2 = [*row2, dog_results[0]['bbox'].tolist()]
                        wr2.writerow(row2)
                    else:
                        mmdet_results_list.append(dog_results)
                        wr2.writerow(row2)
                index = 0
                len_img_list = len(img_list)

                # calculate images numbers
                for image in img_list:
                    if image[-1] != 'g':
                        len_img_list -= 1

                for image in img_list:
                    if image[-1] != 'g':
                        continue
                    print(image)
                    img = args.image_root + '/' + folder + '/' + folder2 + '/' + image

                    # interpolate bbox value
                    if index != 0 and mmdet_results_list[index] == [] and index < len_img_list - 1:
                        prev_i = index - 1
                        next_i = index + 1

                        while mmdet_results_list[prev_i] == [] and prev_i > 0:
                            prev_i -= 1

                        while mmdet_results_list[next_i] == [] and next_i < len_img_list - 1:
                            next_i += 1

                        try:
                            if mmdet_results_list[prev_i] != [] and mmdet_results_list[next_i] != []:
                                mmdet_results_list[index].append({'bbox': (mmdet_results_list[prev_i][0]['bbox'] +
                                                                           mmdet_results_list[next_i][0]['bbox']) / 2})
                            print('no object detection result:', image, index, prev_i, next_i)
                        except:
                            print('except:', image)

                    start = time.time()
                    # test a single image, with a list of bboxes.
                    pose_results, returned_outputs = inference_top_down_pose_model(
                        pose_model,
                        img,
                        mmdet_results_list[index],
                        bbox_thr=0.01,
                        format='xyxy',
                        dataset=pose_model.cfg.data.test.type,
                        return_heatmap=False,
                        outputs=None)

                    pose = pose_estimator.inference(img, mmdet_results_list[index][0]['bbox'])
                    onnx_pose_result = mmdet_results_list[index][0].copy()
                    onnx_pose_result['keypoints'] = pose
                    onnx_pose_result['bbox'] = np.array(mmdet_results_list[index][0]['bbox'])
                    onnx_pose_results = [onnx_pose_result]

                    time_pe.append(time.time() - start)
                    #                 print('-PE',image, ':', time_pe[-1])

                    # save pose results
                    if pose_results != []:
                        row = [image]
                        for i in pose_results[0]['keypoints']:
                            row = [*row, *i.tolist()]
                        wr.writerow(row)
                    else:
                        wr.writerow([image])

                    if onnx_pose_result != []:
                        row = [image]
                        for i in onnx_pose_result[0]['keypoints']:
                            row = [*row, *i.tolist()]
                        wr.writerow(row)
                    else:
                        wr.writerow([image])
                    # show the results
                    if args.visualize:
                        vis_img = vis_pose_result(
                            pose_model,
                            img,
                            pose_results,
                            kpt_score_thr=0.01,
                            dataset=pose_model.cfg.data.test.type,
                            show=False)

                        onnx_vis_img = vis_pose_result(
                            pose_model,
                            img,
                            onnx_pose_results,
                            kpt_score_thr=0.01,
                            dataset=pose_model.cfg.data.test.type,
                            show=False)

                        cv2.imwrite(args.result_dir + '/' + folder + '/' + folder2 + '/vis_' + image, vis_img)
                        cv2.imwrite(args.result_dir + '/' + folder + '/' + folder2 + '/vis_onnx_' + image, onnx_vis_img)
                    index += 1

            shutil.copy(args.result_dir + '/' + folder + '/' + folder2 + '/vis_' + folder2 + '.csv',
                        result_csv_folder + '/' + folder + '/' + folder2 + '/CollectedData_teamDLC.csv')

            shutil.copy(args.result_dir + '/' + folder + '/' + folder2 + '/vis_onnx_' + folder2 + '.csv',
                        onnx_result_csv_folder + '/' + folder + '/' + folder2 + '/CollectedData_teamDLC.csv')

    # print(time_od, '\n')
    # print(time_pe, '\n')
    print('inference_time_od:', sum(time_od) / len(time_od), " inference_time_pe:", sum(time_pe) / len(time_pe))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to ONNX')
    parser.add_argument('--image_root', default='./data/val/dog_pose_track/', type=str)
    parser.add_argument('--result_dir', default='./infer_results', type=str)
    parser.add_argument('--pose_checkpoint', default='./best_AP_epoch_40.pth', type=str)
    parser.add_argument('--pose_config',
                        default='./configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_192x256_fp16.py',
                        type=str)
    parser.add_argument('--onnx_path', default='./hrnet_w32_192x256_fp16_e40.onnx', type=str)
    parser.add_argument('--det_checkpoint',
                        default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
                        type=str)
    parser.add_argument('--det_config', default='./demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', type=str)
    parser.add_argument('--visualize', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
