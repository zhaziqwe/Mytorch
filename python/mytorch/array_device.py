import cupy

class Device():
    pass

class CPUDevice(Device):
    def __repr__(self):
        return "mytorch.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

def cpu():
    return CPUDevice()


class GPUDevice(Device):
    def __repr__(self):
        return "mytorch.gpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, GPUDevice)

    def enabled(self):
        if cupy.cuda.runtime.getDeviceCount() > 0:
            print("GPU is available")
        else:
            print("GPU is not available")

def gpu():
    return GPUDevice()

