import argparse
from onnx_pose_estimator import OnnxPoseEstimator
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import glob
import os
from PIL import Image
import cv2


def main(args):
    pose_estimator_onnx = OnnxPoseEstimator(pose_config=args.pose_config, onnx_path=args.onnx_path)
    gpu_device = 'cuda'
    pose_estimator_pytorch = init_pose_model(args.pose_config, args.pose_checkpoint, device=gpu_device)
    os.makedirs(args.result_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(args.image_root, "*")):
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        ori_im = Image.open(file_name).convert("RGB")
        ori_w, ori_h = ori_im.size
        for i in range(args.resize_cnt):
            target_w = round(ori_w - ((args.resize_ratio * i) * ori_w))
            target_h = round(ori_h - ((args.resize_ratio * i) * ori_h))
            resize_im = ori_im.resize((target_w, target_h), Image.ANTIALIAS)
            dog_bbox = [0., 0., float(target_w), float(target_h), 0.9]
            im_array = np.array(resize_im)
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_estimator_pytorch,
                im_array,
                [{'bbox': dog_bbox}],
                bbox_thr=0.01,
                format='xyxy',
                dataset=pose_estimator_pytorch.cfg.data.test.type,
                return_heatmap=False,
                outputs=None)
            pose = pose_estimator_onnx.inference(im_array, dog_bbox)

            onnx_pose_result = {}
            onnx_pose_result['keypoints'] = pose
            onnx_pose_result['bbox'] = np.array(dog_bbox)
            onnx_pose_results = [onnx_pose_result]

            # show the results
            if args.visualize:
                vis_img = vis_pose_result(
                    pose_estimator_pytorch,
                    im_array,
                    pose_results,
                    kpt_score_thr=0.01,
                    dataset=pose_estimator_pytorch.cfg.data.test.type,
                    show=False)

                onnx_vis_img = vis_pose_result(
                    pose_estimator_pytorch,
                    im_array,
                    onnx_pose_results,
                    kpt_score_thr=0.01,
                    dataset=pose_estimator_pytorch.cfg.data.test.type,
                    show=False)

                cv2.imwrite(os.path.join(args.result_dir, file_name + "_{}x{}_pytorch.jpg".format(target_w, target_h)),
                            vis_img)
                cv2.imwrite(os.path.join(args.result_dir, file_name + "_{}x{}_onnx.jpg".format(target_w, target_h)),
                            onnx_vis_img)


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
    parser.add_argument('--resize_ratio', default=0.05, type=float)
    parser.add_argument('--resize_cnt', default=15, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
