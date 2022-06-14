import onnx
import onnxruntime as rt
from mmpose.apis import init_pose_model
import numpy as np
import torch
import time
import argparse
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
from mmpose.core.bbox import bbox_xywh2cs
from mmpose.datasets.pipelines import Compose

def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def main(args):
    # config = "configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_256x256.py"
    # checkpoint = "best_ar75_e60.pth"
    # onnx_path = "best_hrnet_w32_256x256_e60.onnx"
    # # onnx_path = "best_hrnet_w32_192x256_e60.onnx"
    device = "cpu"

    model = init_pose_model(args.config, args.checkpoint, device=device)
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    model.cpu().eval()

    one_img = torch.randn(args.shape)

    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)

    # check the numerical value
    # get pytorch output
    pytorch_results = model(one_img)
    print("pytorch result")
    print(pytorch_results)
    if not isinstance(pytorch_results, (list, tuple)):
        assert isinstance(pytorch_results, torch.Tensor)
        pytorch_results = [pytorch_results]

    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='TopDownGetBboxCenterScale', padding=1.25),
        dict(type='TopDownAffine'),
        dict(type='ToTensor'),
        dict(
            type='NormalizeTensor',
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'image_file', 'center', 'scale', 'rotation', 'bbox_score',
                'flip_pairs'
            ]),
    ]
    test_pipeline = Compose(test_pipeline)

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

    center, scale = bbox_xywh2cs(
        bbox,
        aspect_ratio=aspect_ratio,
        padding=self.padding,
        pixel_std=self.pixel_std)

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





    # get onnx output
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    sess = rt.InferenceSession(args.onnx_path)
    onnx_input = one_img.detach().numpy()
    onnx_results = sess.run(None, {net_feed_input[0]: onnx_input})
    print("onnx result")
    print(onnx_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to ONNX')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--onnx_path', type=str, default='tmp.onnx')
    parser.add_argument('--test_cnt', type=int, default=100)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 256, 192],
        help='input size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
