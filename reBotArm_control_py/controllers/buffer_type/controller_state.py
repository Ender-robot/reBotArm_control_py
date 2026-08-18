import enum

class ControllerState(enum.Enum):
    """ 控制器占用状态 """
    IDLE = "idle"
    SERVOJ = "servoj"
    SERVOL = "servol"
