import argparse
from onnx_pose_estimator import OnnxPoseEstimator


def main(args):
    pose_estimator = OnnxPoseEstimator(pose_config=args.pose_config, onnx_path=args.onnx_path)
    img_path = "210930_1S_frame_3050.png"
    dog_bbox = [233.68967, 564.35767, 510.79337, 706.0123, 0.98687196]
    pose = pose_estimator.inference(img_path, dog_bbox)
    print("result")
    print(pose)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to ONNX')
    parser.add_argument('--pose_config',
                        default='./configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_192x256_fp16.py',
                        type=str)
    parser.add_argument('--onnx_path', default='./hrnet_w32_192x256_fp16_e40.onnx', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
