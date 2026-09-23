import onnxruntime as ort

session = ort.InferenceSession("w600k_mbf.onnx")

print("Inputs:")

for input_tensor in session.get_inputs():
    print("Name:", input_tensor.name)
    print("Shape:", input_tensor.shape)
    print("Type:", input_tensor.type)

print("\nOutputs:")

for output_tensor in session.get_outputs():
    print("Name:", output_tensor.name)
    print("Shape:", output_tensor.shape)
    print("Type:", output_tensor.type)