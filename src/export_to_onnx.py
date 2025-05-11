import torch
import argparse
from train_pytorch_lstm import AttentionLSTMModel
from train_ratio_predictor import TransformerRatioPredictor

def export_model_total(pt_path, onnx_path, input_size=20):
    model = AttentionLSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    dummy_input = torch.randn(1, 14, input_size)  # [batch, seq_len, features]
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=11)
    print(f"✅ Total model exported to {onnx_path}")

def export_model_ratio(pt_path, onnx_path, input_size=20):
    model = TransformerRatioPredictor(input_size=input_size, embed_dim=64, num_heads=4, ff_hidden_dim=128, output_size=7)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    dummy_input = torch.randn(1, 14, input_size)  # [batch, seq_len, features]
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=11)
    print(f"✅ Ratio model exported to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["total", "ratio"], required=True)
    parser.add_argument("--pt_path", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--input_size", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "total":
        export_model_total(args.pt_path, args.onnx_path, input_size=args.input_size)
    elif args.mode == "ratio":
        export_model_ratio(args.pt_path, args.onnx_path, input_size=args.input_size)
