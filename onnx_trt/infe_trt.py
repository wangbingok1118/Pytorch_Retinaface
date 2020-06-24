#coding=utf-8
import os
import sys
import tensorrt as trt
import cv2
import numpy as np
import time

import pycuda.autoinit
import pycuda.driver as cuda
sys.path.append('..')
from detect_api import postprocess_detection

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

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


ENGINE_FILE = '../weights/trt_engine_onnx_4_1.plan'
BATCH_SIZE = 4  # onnx export input batch size
RESIZE_DIM = (640,640)
output_shape_dict = {
    'box':[BATCH_SIZE,16800,4], # box
    'landmark':[BATCH_SIZE,16800,10], # landmark
    'cls':[BATCH_SIZE,16800,2], # cls
}
device = 'cuda'
save_image = False
vis_thres = 0.6

input_image_path = '../test_images'
output_image_path = '../result_images'

def generate_images_for_engine(batch_images_path_list):
    img_raw_list = []
    img_list = []
    for i_image_path in batch_images_path_list:
        img_raw = cv2.imread(i_image_path,cv2.IMREAD_COLOR)
        img_raw = cv2.resize(img_raw,RESIZE_DIM)
        img = np.float32(img_raw)
        img -= (104, 117, 123)
        img = img.transpose(2,0,1)
        img_list.append(img)
        img_raw_list.append(img_raw)
    img_np_nchw = np.stack(img_list,axis=0)
    return img_np_nchw,img_raw_list

def preInfe(images_folder):
    image_path_list = []
    for i in os.listdir(images_folder):
        if i[0] == '.':
            continue
        image_path_list.append(os.path.join(images_folder,i))
    image_path_list = image_path_list * 100
    print("images num : ",len(image_path_list))
    begin_index = 0
    for end_index in range(0,len(image_path_list),BATCH_SIZE):
        i_batch_list = image_path_list[begin_index:end_index]
        if len(i_batch_list) != BATCH_SIZE:
            continue
        begin_index = end_index
        img_np_nchw,img_raw_list = generate_images_for_engine(i_batch_list)
        yield img_np_nchw,img_raw_list,[i.split('/')[-1] for i in i_batch_list]

def doInfe(context,inputs, outputs, bindings, stream):
    global net_infe_time
    global post_process_time
    for i_img_nchw , i_img_raw_list ,i_img_name_list in preInfe(input_image_path):
        infe_begin_time = time.time()
        inputs[0].host = i_img_nchw.reshape(-1)
        # Transfer data from CPU to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle) # batch_size  is  engine's  max_batch_size
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        output_list = [out.host for out in outputs]
        infe_end_time = time.time()
        net_infe_time += (infe_end_time - infe_begin_time)

        post_begin_time = time.time()
        postInfe(output_list,i_img_raw_list,i_img_name_list)
        post_end_time = time.time()
        post_process_time += (post_end_time - post_begin_time)
    pass

def postInfe(trt_outputs,img_raw,img_name_list):
    # print('processing postProcess trt output :')
    # for i in trt_outputs:
    #     print(i.shape)
    box_batch = trt_outputs[1].reshape(*output_shape_dict['box'])
    land_batch = trt_outputs[0].reshape(*output_shape_dict['landmark'])
    conf_batch = trt_outputs[2].reshape(*output_shape_dict['cls'])

    dets = postprocess_detection(device,box_batch,conf_batch,land_batch)
    for i in range(len(img_name_list)):
        i_dets = dets[i]
        i_img_raw = img_raw[i]
        # print(i_dets.shape)
        # show image
        if save_image:
            for b in i_dets:
                if b[4] < vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(i_img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(i_img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(i_img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(i_img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(i_img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(i_img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(i_img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            name = os.path.join(output_image_path,img_name_list[i])
            cv2.imwrite(name, i_img_raw)

def create_engine(engine_file):
    if os.path.exists(engine_file):
        with open(engine_file,'rb') as f , trt.Runtime(TRT_LOGGER) as runtime:
            return True,runtime.deserialize_cuda_engine(f.read())
    print("{} not exists".format(engine_file))
    return False,None

def create_context(engine_file):
    flag ,engine = create_engine(engine_file)
    if not flag:
        print("ERROR")
        raise Exception
    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings
    return context,inputs, outputs, bindings, stream

def main():
    context,inputs, outputs, bindings, stream = create_context(ENGINE_FILE)
    doInfe(context,inputs, outputs, bindings, stream)
    pass

if __name__ == '__main__':
    net_infe_time = 0.0
    post_process_time = 0.0
    main()
    print("net_infe_time : ",net_infe_time)
    print("post_process_time : ",post_process_time)

