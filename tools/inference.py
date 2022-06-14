import cv2
import argparse
import os, time, csv, time, shutil
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from mmcv.parallel import collate, scatter
import mmcv
import numpy as np
from mmpose.core.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
from mmpose.datasets.pipelines import Compose

import onnx
import onnxruntime as rt
import copy


def decode_heatmap(img_metas, output):
    batch_size = len(img_metas)

    if 'bbox_id' in img_metas[0]:
        bbox_ids = []
    else:
        bbox_ids = None

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    image_paths = []
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']
        image_paths.append(img_metas[i]['image_file'])

        if 'bbox_score' in img_metas[i]:
            score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
        if bbox_ids is not None:
            bbox_ids.append(img_metas[i]['bbox_id'])

    preds, maxvals = keypoints_from_heatmaps(
        output,
        c,
        s,
        unbiased=self.test_cfg.get('unbiased_decoding', False),
        post_process=self.test_cfg.get('post_process', 'default'),
        kernel=self.test_cfg.get('modulate_kernel', 11),
        valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                              0.0546875),
        use_udp=self.test_cfg.get('use_udp', False),
        target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score

    result = {}

    result['preds'] = all_preds
    result['boxes'] = all_boxes
    result['image_paths'] = image_paths
    result['bbox_ids'] = bbox_ids

    return result

def _inference_single_pose_model_onnx(sess,
                                      onnx_input_key,
                                      imgs_or_paths,
                                      bbox,
                                      dataset='TopDownCocoDataset',
                                      cfg=None):
    _test_pipeline = copy.deepcopy(cfg.test_pipeline)

    has_bbox_xywh2cs = False
    for transform in _test_pipeline:
        if transform['type'] == 'TopDownGetBboxCenterScale':
            has_bbox_xywh2cs = True
            break
    if not has_bbox_xywh2cs:
        _test_pipeline.insert(
            0, dict(type='TopDownGetBboxCenterScale', padding=1.25))
    test_pipeline = Compose(_test_pipeline)

    flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                  [16, 17], [18, 19]]

    dataset_name = dataset

    data = {
        'bbox':
            bbox,
        'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
        'bbox_id':
            0,  # need to be assigned if batch_size > 1
        'dataset':
            dataset_name,
        'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'rotation':
            0,
        'ann_info': {
            'image_size': np.array(cfg.data_cfg['image_size']),
            'num_joints': cfg.data_cfg['num_joints'],
            'flip_pairs': flip_pairs
        }
    }

    if isinstance(imgs_or_paths, np.ndarray):
        data['img'] = imgs_or_paths
    else:
        data['image_file'] = imgs_or_paths
    data = test_pipeline(data)
    # batch_data = [data]
    # batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    # batch_data = scatter(batch_data, [device])[0]

    img_meta = data['img_metas'].data
    print(img_meta)
    print(data['img'].shape)
    print(data['img'])
    img_flipped = data['img'].flip(3)
    output_heatmap = sess.run(None, {onnx_input_key: data['img'].detach().numpy()})
    output_flipped_heatmap = sess.run(None, {onnx_input_key: img_flipped.detach().numpy()})
    output_heatmap = (output_heatmap +
                      output_flipped_heatmap) * 0.5

    print(output_heatmap.shape)

    decode_heatmap(img_meta, output_heatmap)
    keypoint_result = self.keypoint_head.decode(
        img_metas, output_heatmap, img_size=[img_width, img_height])


#     batch_data.append(data)

# #     onnx_results = sess.run(None, {onnx_input_key: onnx_input})

#     def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
#         """Defines the computation performed at every call when testing."""
#         assert img.size(0) == len(img_metas)
#         batch_size, _, img_height, img_width = img.shape
#         if batch_size > 1:
#             assert 'bbox_id' in img_metas[0]

#         result = {}

#         features = self.backbone(img)
#         if self.with_neck:
#             features = self.neck(features)
#         if self.with_keypoint:
#             output_heatmap = self.keypoint_head.inference_model(
#                 features, flip_pairs=None)

#         if self.test_cfg.get('flip_test', True):
#             img_flipped = img.flip(3)
#             features_flipped = self.backbone(img_flipped)
#             if self.with_neck:
#                 features_flipped = self.neck(features_flipped)
#             if self.with_keypoint:
#                 output_flipped_heatmap = self.keypoint_head.inference_model(
#                     features_flipped, img_metas[0]['flip_pairs'])
#                 output_heatmap = (output_heatmap +
#                                   output_flipped_heatmap) * 0.5

#         if self.with_keypoint:
#             keypoint_result = self.keypoint_head.decode(
#                 img_metas, output_heatmap, img_size=[img_width, img_height])
#             result.update(keypoint_result)

#             if not return_heatmap:
#                 output_heatmap = None

#             result['output_heatmap'] = output_heatmap

#         return result

#     with torch.no_grad():
#         result = model(
#             img=batch_data['img'],
#             img_metas=batch_data['img_metas'],
#             return_loss=False,
#             return_heatmap=return_heatmap)

#     return result['preds'], result['output_heatmap']

def inference_top_down_pose_model_onnx(sess,
                                       onnx_input_key,
                                       imgs_or_paths,
                                       person_results=None,
                                       bbox_thr=None,
                                       cfg=None):
    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        person_results = [person_results[i] for i in valid_idx]
    if len(bboxes) < 1:
        return None

    bboxes_xyxy = bboxes
    bboxes_xywh = bbox_xyxy2xywh(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    dataset = cfg.data.test.type
    poses = _inference_single_pose_model_onnx(sess,
                                              onnx_input_key,
                                              imgs_or_paths,
                                              bboxes_xywh[0],
                                              dataset=dataset,
                                              cfg=cfg)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    pose_results = []
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results


def main(args):
    # saved csv only folder path of inference images (same data in args.result_dir)
    result_csv_folder = args.result_dir + '_csvOnly'

    gpu_device = 'cuda'

    config = mmcv.Config.fromfile(args.pose_config)

    # initialize pose model
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=gpu_device)
    # initialize detector
    det_model = init_detector(args.det_config, args.det_checkpoint, device=gpu_device)

    onnx_model = onnx.load(args.onnx_path)
    print("loaded onnx model", type(onnx_model))

    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    del onnx_model
    onnx_model = None
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = rt.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
    onnx_input_key = net_feed_input[0]

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(result_csv_folder, exist_ok=True)

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

        for folder2 in os.listdir(args.image_root + '/' + folder):
            if folder2[0] == '.' or folder2[-1] == 's' or folder[-4:] == '.zip':  # skip strange folder
                continue

            print('==', folder, "/", folder2)
            os.makedirs(args.result_dir + '/' + folder + '/' + folder2, exist_ok=True)
            os.makedirs(result_csv_folder + '/' + folder + '/' + folder2, exist_ok=True)

            with open(args.result_dir + '/' + folder + '/' + folder2 + '/vis_' + folder2 + '.csv', 'w',
                      newline='') as w, open(
                args.result_dir + '/' + folder + '/' + folder2 + '/bbox_' + folder2 + '.csv',
                'w', newline='') as w2:
                wr = csv.writer(w)
                wr.writerow(scorer_index)
                wr.writerow(bodyparts_index)
                wr.writerow(coords_index)

                wr2 = csv.writer(w2)
                wr2.writerow(bodyparts_index2)
                wr2.writerow(coords_index2)

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
                    break

                index = 0
                len_img_list = len(img_list)

                # calculate images numbers
                for image in img_list:
                    if image[-1] != 'g':
                        len_img_list -= 1

                for image in img_list:
                    if image[-1] != 'g':
                        continue

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
                    print("bbox result")
                    print(mmdet_results_list[index])
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

                    onnx_pose_results = inference_top_down_pose_model_onnx(sess, onnx_input_key,
                                                                           img,
                                                                           mmdet_results_list[index],
                                                                           bbox_thr=0.01,
                                                                           cfg=config)
                    sys.exit()

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

                    # show the results
                    vis_img = vis_pose_result(
                        pose_model,
                        img,
                        pose_results,
                        kpt_score_thr=0.01,
                        dataset=pose_model.cfg.data.test.type,
                        show=False)

                    index += 1
                    cv2.imwrite(args.result_dir + '/' + folder + '/' + folder2 + '/vis_' + image, vis_img)

            shutil.copy(args.result_dir + '/' + folder + '/' + folder2 + '/vis_' + folder2 + '.csv',
                        result_csv_folder + '/' + folder + '/' + folder2 + '/CollectedData_teamDLC.csv')

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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
