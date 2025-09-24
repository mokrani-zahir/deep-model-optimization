import os
import argparse
import urllib.request

DEFAULT_MODELS_DIR = "./models"

MODEL_URLS = {
    "craft_pytorch": "https://huggingface.co/zahir-mokrani/ocr_detect_text/resolve/main/craft_mlt_25k.pth",
    "craft_onnx": "https://huggingface.co/<user>/<repo>/resolve/main/craft.onnx",
    "craft_openvino_xml": "https://huggingface.co/<user>/<repo>/resolve/main/craft.xml",
    "craft_openvino_bin": "https://huggingface.co/<user>/<repo>/resolve/main/craft.bin",
}

def download_model(name, output_dir=DEFAULT_MODELS_DIR):
    """Download a single model by name."""
    if name not in MODEL_URLS:
        print(f"[ERROR] Model '{name}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    url = MODEL_URLS[name]
    filename = os.path.join(output_dir, os.path.basename(url))

    if not os.path.exists(filename):
        print(f"Downloading {name} from {url} ...")
        urllib.request.urlretrieve(url, filename)
        print(f"[OK] {name} saved to {filename}")
    else:
        print(f"[SKIP] {filename} already exists.")

def main():
    parser = argparse.ArgumentParser(description="Download pretrained models for CRAFT")
    parser.add_argument(
        "--model", type=str, default="craft_pytorch",
        help="Model name to download (choices: craft_pytorch, craft_onnx, craft_openvino_xml, craft_openvino_bin, all)"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_MODELS_DIR,
        help="Output directory where models will be saved"
    )

    args = parser.parse_args()

    if args.model == "all":
        for name in MODEL_URLS:
            download_model(name, args.output)
    else:
        download_model(args.model, args.output)

if __name__ == "__main__":
    main()
