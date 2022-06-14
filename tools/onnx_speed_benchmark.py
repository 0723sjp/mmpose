import onnx
import onnxruntime as rt
from mmpose.apis import init_pose_model
import numpy as np
import torch
import time
import argparse


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

    pytorch_total = 0.
    for i in range(args.test_cnt):
        print("pytorch test", i)
        start = time.time()
        model(one_img)
        exec_time = time.time() - start
        print(exec_time)
        pytorch_total += exec_time

    print("completed pytorch test")

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

    onnx_total = 0.
    for i in range(args.test_cnt):
        print("onnx test", i)
        start = time.time()
        sess.run(None, {net_feed_input[0]: onnx_input})
        exec_time = time.time() - start
        print(exec_time)
        onnx_total += exec_time

    # compare results
    assert len(pytorch_results) == len(onnx_results)

    print("onnx", onnx_total / args.test_cnt)
    print("pytorch", pytorch_total / args.test_cnt)

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