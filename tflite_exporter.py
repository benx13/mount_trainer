import torch
from mcunet.model_zoo import net_id_list, build_model
import os
import tensorflow as tf
import numpy as np
import onnx
from onnx_tf.backend import prepare

print(net_id_list)  # the list of models in the model zoo

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-tiny", pretrained=False)  # you can replace net_id with any other option from net_id_list

print(image_size, description)

print(model)

# Create export directory if it doesn't exist
export_dir = "tflite_export"
os.makedirs(export_dir, exist_ok=True)

# Prepare dummy input tensor
batch_size = 1
channels = 3  # Assuming RGB input
dummy_input = torch.randn(batch_size, channels, image_size, image_size)

# Export to ONNX
onnx_path = os.path.join(export_dir, "mcunet_tiny.onnx")
torch.onnx.export(
    model,                     # PyTorch model
    dummy_input,              # Input tensor
    onnx_path,               # Output file path
    export_params=True,      # Store the trained weights inside the model file
    opset_version=11,        # ONNX version to use
    do_constant_folding=True,# Whether to execute constant folding for optimization
    input_names=['input'],   # Name of the input tensor
    output_names=['output'], # Name of the output tensor
    dynamic_axes={           # Variable length axes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported to {onnx_path}")

# Convert ONNX to TensorFlow SavedModel format
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_model_path = os.path.join(export_dir, "mcunet_tiny_tf")
tf_rep.export_graph(tf_model_path)

# Convert to TFLite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Define representative dataset generator for calibration
def representative_dataset():
    for _ in range(100):  # Number of calibration samples
        data = torch.randn(1, 3, image_size, image_size)
        yield [data.numpy().astype(np.float32)]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

# Save TFLite model
tflite_path = os.path.join(export_dir, "mcunet_tiny_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"INT8 quantized model exported to {tflite_path}")