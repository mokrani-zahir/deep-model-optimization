import torch
import torch.onnx
from src.craft import CRAFT
from collections import OrderedDict

def convert_pth_to_onnx(pth_path, onnx_path, input_size=(1280, 1280), opset_version=13):

    def copy_state_dict(state_dict):
        """Fix state_dict keys if saved from DataParallel"""
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    # Load PyTorch model
    net = CRAFT()
    net.load_state_dict(copy_state_dict(torch.load(pth_path, map_location='cpu')))
    net.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, *input_size)

    # Export ONNX
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output', 'features'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
            'features': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"ONNX model exported: {onnx_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CRAFT PyTorch model (.pth) to ONNX")
    parser.add_argument("--source", default="./models/craft_mlt_25k.pth", help="Path to .pth file")
    parser.add_argument("--output", default="./models/craft_mlt_25k.onnx", help="Path to save .onnx file")
    parser.add_argument("--height", type=int, default=1280, help="Input image height")
    parser.add_argument("--width", type=int, default=1280, help="Input image width")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    args = parser.parse_args()

    convert_pth_to_onnx(
        pth_path=args.source,
        onnx_path=args.output,
        input_size=(args.height, args.width),
        opset_version=args.opset
    )
