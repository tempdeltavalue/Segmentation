from model import Deeplabv3
import coremltools
import tensorflow_model_optimization as tfmot
import tensorflow as tf

import numpy as np
import keras2onnx
from coremltools.proto import NeuralNetwork_pb2

from custom_model import Deeplabv3Model

BASE_END_POINT = "converted_models"
ONNX_PATH = BASE_END_POINT + '/model.onnx'


# The conversion function for Lambda layers.
def convert_lambda(layer):
    print("layer_function", layer.function)
    # Only convert this Lambda layer if it is for our swish function.
    if layer.function == "VOVA":
        params = NeuralNetwork_pb2.CustomLayerParams()

        # The name of the Swift or Obj-C class that implements this layer.
        params.className = "Swish"

        # The desciption is shown in Xcode's mlmodel viewer.
        params.description = "A fancy new activation function"

        # Set configuration parameters
        # params.parameters["someNumber"].intValue = 100
        # params.parameters["someString"].stringValue = "Hello, world!"

        # Add some random weights
        # my_weights = params.weights.add()
        # my_weights.floatValue.extend(np.random.randn(10).astype(float))

        return params
    else:
        return None

def convert_to_onnx():
    model = Deeplabv3()

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model,
                                          model.name,
                                          channel_first_inputs=['input_1'])
    keras2onnx.save_model(onnx_model, ONNX_PATH)

import keras
from coremltools.models.neural_network import quantization_utils

def convert_to_coreml():
    # deeplab_model = onnx.load(ONNX_PATH)
    # current_converter = coremltools.converters.onnx
    # coreml_model = coremltools.converters.onnx.convert(deeplab_model,
    #                                                    image_input_names="input_1",
    #                                                    minimum_ios_deployment_target='13')

    deeplab_model = Deeplabv3()
    # deeplab_model = tfmot.quantization.keras.quantize_model(deeplab_model)
    current_converter = coremltools.converters.keras
    coreml_model = coremltools.convert(deeplab_model,
                                                        inputs=[coremltools.ImageType()],
                                                        output_names=["output_layer"],
                                                        image_input_names="input_1",
                                                        # source="tensorflow",
                                                        # image_scale=2/255.0,
                                                        # red_bias=-1,
                                                        # green_bias=-1,
                                                        # blue_bias=-1,

                                             # add_custom_layers=True,
                                             # custom_conversion_functions = {"Lambda": convert_lambda}
    )

    spec = coreml_model.get_spec()
    # spec.neuralNetwork.preprocessing[0].featureName = 'input_1'

    print(spec.neuralNetwork.preprocessing)

    # quan_model = quantization_utils.quantize_weights(coreml_model, 8)
    # quan_model = coremltools.models.MLModel(quan_model)

    # input = coreml_model.get_spec().input[0]
    # spec = coreml_model._spec
    # spec.description.input[0].type.multiArrayType.shape.extend([3, 1, 150])
    # coremltools.util.save_spec(spec, "YourNewModel.mlmodel")
    # print("get_spec", coreml_model.get_spec())
    coreml_model.predict()
    coreml_model.save(BASE_END_POINT + '/keras_deeplab_v3.mlmodel')
    print("coreml_model", coreml_model)




if __name__ ==  "__main__":
    # convert_to_onnx()
    convert_to_coreml()
    # frozen_keras_graph(r"C:\Users\m\Desktop\Image Segmentation", "Vova")