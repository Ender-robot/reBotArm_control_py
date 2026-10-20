"""两种通讯器共用的周期与线程生命周期。"""

import threading
import time


class CommunicationBase:
    def __init__(self, hardware):
        self.hardware = hardware
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, name="rebotarm-communication", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self):
        next_cycle = time.monotonic()
        while not self.stop_event.is_set():
            self.hardware.update_feedback()
            self.step()
            next_cycle += self.hardware._comm_period
            timeout = next_cycle - time.monotonic()
            if timeout > 0:
                self.stop_event.wait(timeout)
            else:
                next_cycle = time.monotonic()

    def step(self):
        raise NotImplementedError

    def reset(self, reason):
        pass
