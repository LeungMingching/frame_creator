import numpy as np


class NaviLayer:

    def __init__(self) -> None:
        self.navi_command = None
        self.execute_distance = None

    def create_navi(self, navi_command, execute_distance):
        self.navi_command = navi_command
        self.execute_distance = execute_distance


if __name__ == '__main__':
    nl = NaviLayer()
    nl.create_navi()