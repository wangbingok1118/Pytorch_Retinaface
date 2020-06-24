#coding=utf-8
import os
import sys
import tensorrt as trt
import cv2
import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine
max_batch_size = 1 # the max_batch_size : The maximum batch size which can be used at execution time
# batch size is control by onnx input

def create_engine(onnx_file,engine_file,max_batch_size=max_batch_size):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network( flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) ) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = 1 << 30  # Your workspace size
        builder.max_batch_size = max_batch_size
        # Parse model file
        if not os.path.exists(onnx_file):
            quit('ONNX file {} not found'.format(onnx_file))

        print('Loading ONNX file from path {}...'.format(onnx_file))
        with open(onnx_file, 'rb') as model:
            print('Beginning ONNX file parsing')
            parse_flag = parser.parse(model.read())
            if not parse_flag:
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file))

        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        with open(engine_file, "wb") as f:
            f.write(engine.serialize())
        return engine

def get_engine(onnx_file,engine_file):
    # if os.path.exists(engine_file):
    #     print("engine file already exists : {}".format(engine_file))
    #     with open(engine_file,'rb') as f,trt.Runtime(TRT_LOGGER) as runtime:
    #         return runtime.deserialize_cuda_engine(f.read())
    engine = create_engine(onnx_file,engine_file)
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def main():
    onnx_file = '../weights/FaceDetector_4.onnx'
    engine_file = '../weights/trt_engine_onnx_4_{}.plan'.format(max_batch_size)
    engine = get_engine(onnx_file,engine_file)


if __name__ == '__main__':
    main()

