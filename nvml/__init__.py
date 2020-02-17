import numpy as np
import platform
import threading as th
import time
from collections import OrderedDict, defaultdict, deque
from py3nvml import py3nvml


def _parse_samples(samples, val='uiVal'):
    return [(s.timeStamp, getattr(s.sampleValue, val)) for s in samples[1]]


boot_time = int(time.time() * 1000)


class GPU(object):
    def __init__(self, pos):
        self.pos = pos
        self.handle = py3nvml.nvmlDeviceGetHandleByIndex(pos)
        self.name = py3nvml.nvmlDeviceGetName(self.handle)
        self._tm = None
        self.last_sampled = defaultdict(lambda: boot_time)
        self.is_virtual = False  # For limiting usage of commands
        self.arch = None

    def json(self):
        return {
            "name": self.name,
            "is_virtual": self.is_virtual,
            "clocks": self._get_clock(),
            "util": self._get_utilisation(),
            "memory": self.memory,
            "pci_info": self.pci_info
        }

    def _get_memory(self):
        return py3nvml.nvmlDeviceGetMemoryInfo(self.handle)

    def _get_clock(self):
        return {
            'GRPAHICS': py3nvml.nvmlDeviceGetClockInfo(self.handle, py3nvml.nvmlClockType_t.NVML_CLOCK_GRAPHICS.value),
            'SM': py3nvml.nvmlDeviceGetClockInfo(self.handle, py3nvml.nvmlClockType_t.NVML_CLOCK_SM.value),
            'MEM': py3nvml.nvmlDeviceGetClockInfo(self.handle, py3nvml.nvmlClockType_t.NVML_CLOCK_MEM.value),
            'VIDEO': py3nvml.nvmlDeviceGetClockInfo(self.handle, py3nvml.nvmlClockType_t.NVML_CLOCK_VIDEO.value)
        }

    def _get_power(self):
        return py3nvml.nvmlDeviceGetPowerUsage(self.handle)

    def _get_utilisation(self):
        res = py3nvml.nvmlDeviceGetUtilizationRates(self.handle)
        return {
            'gpu': res.gpu,
            'memory': res.memory
        }

    def _get_power_management_limit(self):
        return py3nvml.nvmlDeviceGetPowerManagementLimit(self.handle)

    def _get_power_management_limit_constraints(self):
        return py3nvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handle)

    def _get_temperature(self):
        return py3nvml.nvmlDeviceGetTemperature(self.handle,
                                                py3nvml.nvmlTemperatureSensors_t.NVML_TEMPERATURE_GPU.value)

    def _get_fan_speed(self):
        return py3nvml.nvmlDeviceGetFanSpeed(self.handle)

    def _get_running_graphic_processes(self):
        return [(p.pid, p.usedGpuMemory >> 20) for p in py3nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)]

    def _get_running_compute_processes(self):
        return [(p.pid, p.usedGpuMemory >> 20) for p in py3nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)]

    def _get_pci_info(self):
        return py3nvml.nvmlDeviceGetPciInfo(self.handle)

    def _get_pci_throughput(self):
        return {
            'rx': py3nvml.nvmlDeviceGetPcieThroughput(self.handle,
                                                      py3nvml.nvmlPcieUtilCouter_t.NVML_PCIE_UTIL_RX_BYTES.value),
            'tx': py3nvml.nvmlDeviceGetPcieThroughput(self.handle,
                                                      py3nvml.nvmlPcieUtilCouter_t.NVML_PCIE_UTIL_TX_BYTES.value)
        }

    def _get_samples(self, sample_type):
        try:
            samples = _parse_samples(
                py3nvml.nvmlDeviceGetSamples(self.handle, sample_type, self.last_sampled[sample_type]), 'uiVal')
            samples.sort(key=lambda x: x[0])
            self.last_sampled[sample_type] = samples[-1][0]
        except py3nvml.NVMLError:
            samples = []
        return samples

    def sample_power(self):
        return self._get_samples(0)

    def sample_gpu(self):
        return self._get_samples(1)

    def sample_memory(self):
        return self._get_samples(2)

    def sample_enc(self):
        return self._get_samples(3)

    def sample_dec(self):
        return self._get_samples(4)

    def sample_gpu_clock(self):
        return self._get_samples(5)

    def sample_memory_clock(self):
        return self._get_samples(6)

    @property
    def memory(self):
        mem = self._get_memory()
        return {
            'total': mem.total >> 20,
            'used': mem.used >> 20,
            'free': mem.free >> 20
        }

    @property
    def pci_info(self):
        pci = self._get_pci_info()
        return OrderedDict([
            ('bus', str(pci.bus)),
            ('busId', pci.busId.decode()),
            ('device', str(pci.device)),
            ('domain', str(pci.domain)),
            ('pciDeviceId', str(pci.pciDeviceId)),
            ('pciSubSystemId', str(pci.pciSubSystemId)),
            ('reserved0', str(pci.reserved0)),
            ('reserved1', str(pci.reserved1)),
            ('reserved2', str(pci.reserved2)),
            ('reserved3', str(pci.reserved3)),
        ])

    @property
    def total_memory(self):
        if self._tm is None:
            self._tm = self._get_memory().total
        return self._tm

    @property
    def free_memory(self):
        return self._get_memory().free

    def __repr__(self):
        return "%s<%s|%s, total=%sMiB, free=%sMiB>" % (
        self.__class__.__name__, self.pos, self.name, self.total_memory >> 20, self.free_memory >> 20,)


class System(object):
    def __init__(self):
        py3nvml.nvmlInit()
        self.driver = py3nvml.nvmlSystemGetDriverVersion()
        deviceCount = py3nvml.nvmlDeviceGetCount()
        self.gpus = [GPU(i) for i in range(deviceCount)]

    def json(self):
        info = platform.uname()
        return {
            "platform": {
                "system": info.system,
                "node": info.node,
                "release": info.release,
                "version": info.version,
                "machine": info.machine,
                "processor": info.processor,
            },
            "nvdriver": self.driver,
            "gpus": {i: gpu.json() for i, gpu in enumerate(self.gpus)}
        }


class SamplerThread(th.Thread):
    def __init__(self, system, maxlen=10000):
        super(SamplerThread, self).__init__()
        self.system = system
        self._lock = th.Lock()
        self._interupt = th.Event()

        self.sample_attrs = ['power', 'gpu', 'memory', 'enc', 'dec', 'temperature', 'fan_speed']

        def make_queue():
            return deque(maxlen=maxlen)

        self.power_hist = defaultdict(make_queue)
        self.gpu_hist = defaultdict(make_queue)
        self.memory_hist = defaultdict(make_queue)
        self.enc_hist = defaultdict(make_queue)
        self.dec_hist = defaultdict(make_queue)
        self.gpu_clock_hist = defaultdict(make_queue)
        self.memory_clock_hist = defaultdict(make_queue)
        self.fan_speed_hist = defaultdict(make_queue)
        self.temperature_hist = defaultdict(make_queue)

    def run(self) -> None:
        while not self._interupt.is_set():
            with self._lock:
                now = int(round(time.time() * 1e6))
                for gpu in self.system.gpus:
                    self.power_hist[gpu].extend(gpu.sample_power())
                    self.gpu_hist[gpu].extend(gpu.sample_gpu())
                    self.memory_hist[gpu].extend(gpu.sample_memory())
                    self.enc_hist[gpu].extend(gpu.sample_enc())
                    self.dec_hist[gpu].extend(gpu.sample_dec())
                    # self.gpu_clock_hist[gpu].extend(gpu.sample_gpu_clock())
                    # self.memory_clock_hist[gpu].extend(gpu.sample_memory_clock())
                    self.fan_speed_hist[gpu].append((now, gpu._get_fan_speed()))
                    self.temperature_hist[gpu].append((now, gpu._get_temperature()))
            time.sleep(1)

    def stop(self) -> None:
        self._interupt.set()
        self.join()

    def _get_gpus(self, attr, of=None):
        with self._lock:
            hist = {}
            for gpu in self.system.gpus:
                v = np.array(list(getattr(self, attr)[gpu]))
                hist[gpu] = v[v[:, 0] > of] if of is not None else v
            return hist

    def get_samples(self, of=None):
        return {s: self._get_gpus('%s_hist' % (s,), of) for s in self.sample_attrs}

    def get_power(self):
        return self._get_gpus('power_hist')

    def get_gpu(self):
        return self._get_gpus('gpu_hist')

    def get_memory(self):
        return self._get_gpus('memory_hist')

    def get_gpu_clock(self):
        return self._get_gpus('gpu_clock_hist')

    def get_memory_clock(self):
        return self._get_gpus('memory_clock_hist')

    def get_enc(self):
        return self._get_gpus('enc_hist')

    def get_dec(self):
        return self._get_gpus('dec_hist')

    def get_temperature(self):
        return self._get_gpus('temperature_hist')

    def get_fan_speed(self):
        return self._get_gpus('fan_speed_hist')


if __name__ == '__main__':
    system = System()
    gpu0 = system.gpus[0]
    gpu1 = system.gpus[1]

    self = gpu0
    print(gpu0)
    st = SamplerThread(system)
    st.start()

    pwr = st.get_power()
    d1 = pwr[system.gpus[1]]
    d0 = pwr[system.gpus[0]]
    ts0 = list(zip(*d0))[0]
    ts1 = list(zip(*d1))[0]

    from datetime import datetime

    str(datetime.fromtimestamp(max(ts0) // 1000000))
    str(datetime.fromtimestamp(max(ts1) // 1000000))
