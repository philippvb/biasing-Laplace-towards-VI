import platform
import os
from datetime import datetime
def infer_root_dir() -> str:
    root_dir = None
    system = platform.system().upper() 
    if system == "LINUX":
        root_dir = "/home/hennig/pvbachmann87/VariationalLaplace"
    elif system == "DARWIN":
        root_dir = "/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/paper"
    else:
        raise KeyError("Couldn't infer system")
    return root_dir

def create_checkpoint_name(run_name:str) -> str:
    checkpoint_name = os.path.join(infer_root_dir(), "run_checkpoints", datetime.strftime(datetime.now(), "%m-%d"), run_name)
    if os.path.exists(checkpoint_name):
        extension = 1
        while os.path.exists(checkpoint_name + "_" + str(extension)):
            extension += 1
        return checkpoint_name +  "_" + str(extension)
    else:
        return checkpoint_name