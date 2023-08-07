"""
This file is used for sending the car control command through the system pipe.
The command is sending through UDP protocol in the same LAN to make sure the real-time control.
"""


import os
import numpy as np
import socket
import time

# set pipe path
read_path = "/tmp/command"
if os.path.exists(read_path):
    os.remove(read_path)
os.mkfifo(read_path)
rf = os.open(read_path, os.O_NONBLOCK | os.O_RDONLY)

# this ip address is the car's network IP
# the port number is the car's network port for receiving control commands
ip_addr = "192.168.117.117"
ip_port = 1234

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
print("UDP server up and listening")


# function to send UDP command, port number 1234
def udp_command_sender(command):
    UDPServerSocket.sendto(str.encode(command), (ip_addr, ip_port))


# [F](forward) [B](backward)[L](turn left)[R](turn right) [S](stop) are the commands for the car
# this function will make sure only these 5 commands will be sent to the car
def to_command(argument):  # default value is [S]
    switcher = {"w": "[F]", "s": "[B]", "a": "[L]", "d": "[R]", "z": "[S]"}
    return switcher.get(argument, "[S]")


# function to read data form the system pipe
def read_pipe_data():
    global key
    try:
        key = os.read(rf, 8)
        key = key.decode("utf-8")

    except OSError as e:
        if e.errno == 11:
            print("wait")
        else:
            print("something wrong")


# read the command and send to car
while True:
    read_pipe_data()
    if key == "exit":
        print("received msg:", key, "terminate.")
        break
    udp_command_sender(to_command(key))

os.close(rf)
