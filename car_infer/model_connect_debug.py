"""
This script handles the connection of the hardware,
passes the data to the model based on 3 second segments,
and the gnerated prediction values 
will be sent to car through car_control_pipeline.py
"""

import sys
import tty
import json
import termios
import select
import socket
import curses
import os
import time
import torch
import numpy as np
import threading
import queue
from distutils.command.config import config
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from new_model import EEGPredictor
from model import XXXPNet_Basic

# from baseline import Tensor_CSPNet


# set pipe path
store = []
write_path = "/tmp/command"
wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

# init hardware
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
# for ubuntu
params.serial_port = "/dev/ttyUSB0"

board = BoardShim(2, params)
board.prepare_session()
board.start_stream()

# set up path for file saving and model reading
absolute_folder_path = (
    "/home/julius/Documents/GitHub/research/EEG_controller_data_record/test_data_draft"
)
model_path = "/home/julius/Documents/GitHub/research/EEG_controller_data_record/EEG/scripts/car_infer/Net/Tensor_CSPNet_model.pth"
model = XXXPNet_Basic()
# model = Tensor_CSPNet(18, True)
# model.load_state_dict(torch.load(model_path))
model.eval()

eeg_predictor = EEGPredictor()
eeg_predictor.set_model(model)

data_queue = queue.Queue()


# functions for saving the recorded data
def save_file(key_data, file_name):
    file_path = os.path.join(absolute_folder_path, file_name)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, "wb") as f:
        np.save(f, key_data, allow_pickle=False, fix_imports=False)


# stash the data stream for batch and send to queue
def collectData(data_queue):
    samples_to_collect = (
        128 * 3
    )  # Collect data for 3 seconds (assuming 128 Hz sampling rate)
    collected_data = []
    while True:
        key = np.array(board.get_board_data())
        key = key[1:17, :]
        collected_data.append(key)
        if len(collected_data) >= samples_to_collect:
            data_queue.put(np.concatenate(collected_data, axis=1))
            collected_data.clear()
        time.sleep(1 / 128)


# check the data queue and do inference if data satisfies
def getData(data_queue):
    while True:
        data = data_queue.get()  # Wait until data is available in the queue
        if data.shape[1] >= 384:
            predict_key(data)


# call the model for inference
# the time part is for inference time cost analysis
def predict_key(key):
    start = time.time()
    prediction = eeg_predictor.predict(key)
    end = time.time()
    print("predict time", end - start)
    store.append(float(end - start))
    file_path = os.path.join(absolute_folder_path, "predict_time.json")
    with open(file_path, "w") as file:
        json.dump(store, file)
    print("prediction:", prediction)
    if prediction == 0:  # move forward
        return "w"
    elif prediction == 1:  # move backward
        return "s"
    elif prediction == 2:  # turn left
        return "a"
    elif prediction == 3:  # turn right
        return "d"
    elif prediction == 4:  # stop
        return "z"


# get the model output and send to pipe
def send_data():
    while True:
        data = data_queue.get()
        print(data)
        prediction = predict_key(data)
        if data is not None:
            print(prediction)
            msg = str(prediction).encode("ascii")
            len_send = os.write(wf, msg)


# execute the code in threads for paralleling the data collection and inference
try:
    collect_data_thread = threading.Thread(target=collectData, args=([data_queue]))
    get_data_thread = threading.Thread(target=getData, args=(data_queue,))

    collect_data_thread.start()
    get_data_thread.start()

    send_data()

finally:
    os.write(wf, "exit".encode("ascii"))
    os.close(wf)
