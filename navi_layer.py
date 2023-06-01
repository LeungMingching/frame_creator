import numpy as np


class NaviLayer:

    def __init__(self) -> None:
        self.navi_command = None
        self.execute_distance = None

    def create_navi(self):
        self.navi_command = np.random.randint(0, 5)
        self.execute_distance = np.random.rand() * 300


if __name__ == '__main__':
    nl = NaviLayer()
    nl.create_navi()