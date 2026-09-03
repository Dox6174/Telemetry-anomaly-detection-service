import torch
import numpy as np
from model import Autoencoder, INPUT_DIM
import onnxruntime as ort

model = Autoencoder()
model.load_state_dict(torch.load("model.pt"))
model.eval()

dummy = torch.randn(1, INPUT_DIM, dtype=torch.float32)

torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["reconstruction"],
    dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
    opset_version=17,
)

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
with torch.no_grad():
    torch_out = model(dummy).numpy()
onnx_out = sess.run(None, {"input": dummy.numpy()})[0]

max_diff = np.abs(torch_out - onnx_out).max()
print(f"Max diff PyTorch vs ONNX: {max_diff:.8f}")
assert max_diff < 1e-4, "ONNX export parity check FAILED — do not proceed until this passes"
print("Parity check passed.")