import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Unified evaluator for CRAFT targets")
    parser.add_argument("--target", choices=["pytorch", "onnx", "openvino"], required=True)
    args, extra_args = parser.parse_known_args()

    # Map backend to Python module
    target_map = {
        "pytorch": "backend.test_pytorch",
        "onnx": "backend.test_onnx",
        "openvino": "backend.test_openvino"
    }

    module_name = target_map[args.target]
    cmd = [sys.executable, "-m", module_name] + extra_args

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
