'''
This file is used for sending the car control command from the keyboard, for testing purpose

press the following keys for corresponding actions:
'w': move forward,
's': move backward,
'a': turn left,
'd': turn right,
'z': stop.
'''


import sys, tty, termios, select, curses
import os, time

# set pipe path
write_path = "/tmp/command"
wf = os.open(write_path, os.O_SYNC | os.O_CREAT | os.O_RDWR)

# get keyboard input
def getKey():
    tty.setraw(sys.stdin.fileno())
    ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
    if ready_to_read:
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))
        # Move the cursor to the front of the line
        curses.setupterm()
        sys.stdout.buffer.write(curses.tigetstr("cr"))
        sys.stdout.flush()
        return key
    else:
        return None

# Initialize curses, this is for beautifying
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()

# read the key input form keayboard and send to pipe
try:
    while True:
        key = getKey()
        if key is not None:
            if key == 'q':
                print("exit!")
                break
            print(key)
            msg = key.encode('ascii')
            # encode could only be used on string
            len_send = os.write(wf, msg)
            print("sent msg: ", msg)
            # time.sleep(1)

finally:
    # Clean up and restore terminal settings
    curses.echo()
    curses.nocbreak()
    curses.endwin()
    os.write(wf, 'exit'.encode('ascii'))
    os.close(wf)
