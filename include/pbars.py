import imp
from global_ import Runtime
from tqdm import tqdm

''' Wrapper of a tqdm progress bar that progress based on simulation time '''
class SimTimeBar:
    def __init__(self, simend, desc, disable) -> None:
        self.simend = simend
        self.pbar = tqdm(total=100, 
                    desc=desc, 
                    disable=disable)
    
    def update(self):
        current = int(100 * Runtime.get().now / self.simend)
        last = self.pbar.n
        self.pbar.update(current - last)

    def close(self):
        self.pbar.close()
