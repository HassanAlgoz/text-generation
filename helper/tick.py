import time
import os
import sys

class Tick():
    def __init__(self, name=os.path.basename(sys.argv[0])):
        self.start = time.time()
        self.name = name
        print("{}...".format(name))
    
    def tock(self):
        end = (time.time() - self.start)
        print("Finished {} in {:.2f}s".format(self.name, end))