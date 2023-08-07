"""
This script records the eeg data from hardware 
and save to files
"""

from distutils.command.config import config
import sys
import tkinter
from PIL import Image, ImageTk
import threading
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import os
import numpy as np
import json
import pickle

# read config from the json file
config_file = open("./config.json")
config_data = json.load(config_file)
data_root = config_data["data_file_root"]
label_root = config_data["label_file_root"]
combined_root = config_data["combined_file_root"]
image_list = config_data["image_list"]
stop_image = config_data["stop_image"]
board_id = config_data["board_id"]
config_file.close()

# read in the current count number of data
file_count = np.loadtxt(fname="./file_count.txt", dtype=float)
file_count = int(file_count)

# prepare the data recording folder
if os.path.exists(data_root):
    print("data_files exists")
else:
    os.mkdir(str(data_root))

if os.path.exists(label_root):
    print("data_files exists")
else:
    os.mkdir(str(label_root))


# init EEG device
BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
# for ubuntu
params.serial_port = "/dev/ttyUSB0"
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()


# This function will present the images for stimulating different EEG signals
def show_image():
    # Definition as global to be controlled out of the function
    global item, canvas

    root = tkinter.Tk()
    root.title("img window")
    root.geometry("1920x1080")
    img = Image.open("./img/white_stop.jpg")
    img = ImageTk.PhotoImage(img)
    canvas = tkinter.Canvas(bg="black", width=1920, height=1080)
    canvas.place(x=0, y=0)
    item = canvas.create_image(0, 0, image=img, anchor=tkinter.NW)
    root.mainloop()


# make a thread and start to display a image
thread1 = threading.Thread(target=show_image)
thread1.start()


# function to save recorded data
def save_data(raw_data, direction_data, data_root, label_root, data_file_name):
    with open(os.path.join(data_root, data_file_name), "wb") as f:
        file1 = np.save(f, raw_data, allow_pickle=False, fix_imports=False)
        with open(os.path.join(label_root, data_file_name), "wb") as f:
            file2 = np.save(f, direction_data)


# show the images and collect data
for j in range(2):
    for a in range(2):
        for i in range(4):
            data_file_name = str(file_count) + ".npy"
            file_count += 1
            img = Image.open(image_list[a][i])
            img = ImageTk.PhotoImage(img)
            canvas.itemconfig(item, image=img)
            # recording time for each image
            time.sleep(25)
            # get label data
            label_data = np.full((1, board.get_board_data_count()), i, dtype=int)
            # get all data and remove it from internal buffer
            raw_data = board.get_board_data()
            # call save data function to save the data
            save_data(raw_data, label_data, data_root, label_root, data_file_name)
            img = Image.open(stop_image[0])
            img = ImageTk.PhotoImage(img)
            canvas.itemconfig(item, image=img)
            # breaking time
            time.sleep(5)
            raw_data = board.get_board_data()


file_count = [file_count]

np.savetxt(fname="./file_count.txt", X=file_count)

board.stop_stream()
board.release_session()

# at the end of data collection present a black screen
img = Image.open(stop_image[1])
img = ImageTk.PhotoImage(img)
canvas.itemconfig(item, image=img)

print("Complete, can close now")
