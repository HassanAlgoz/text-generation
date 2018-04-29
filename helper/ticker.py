import time
import os
import sys

class Ticker():
    """
        A stack-based ticker
        to measure end_time - start_time (in seconds)
        and be able to `name` measured sections.
        
        Usage 1: Pass a name to `.tick` method
           Ticker = Ticker()
           Ticker.tick("section name")
            some code...
           Ticker.tock()
        Output 1:
            section name...
            Finished "section name" in 2.58s



        Usage 2: Passing no parameter `.tick` method
           Ticker = Ticker()
           Ticker.tick()
            some code...
           Ticker.tock()
        Output 2:
            [script-name].py...
            Finished "[script-name].py" in 2.58s
    """

    def __init__(self):
        self.stack = [] # each element is a tuple (time, name)
    
    # Push
    def tick(self, name=os.path.basename(sys.argv[0])):
        self.stack.append((time.time(), name))
        print("{}...".format(name))

    # Pop
    def tock(self):
        t, name = self.stack.pop()
        end = time.time() - t
        print("Finished \"{}\" in {:.2f}s".format(name, end))
