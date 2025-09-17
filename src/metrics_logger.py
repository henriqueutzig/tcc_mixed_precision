# metrics_logger.py
import threading
import time
import statistics

class GpuMetricsLogger:
    """A class to monitor and log GPU metrics in a background thread."""
    def __init__(self, gpu_vendor, gpu_index=0, interval=1):
        if gpu_vendor not in ['nvidia', 'amd']:
            raise ValueError("Unsupported GPU vendor. Choose 'nvidia' or 'amd'.")
        
        self.gpu_vendor = gpu_vendor
        self.gpu_index = gpu_index
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.thread = None
        self.handle = None

        if self.gpu_vendor == 'nvidia':
            try:
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage
                self.nvml = {
                    "init": nvmlInit,
                    "get_handle": nvmlDeviceGetHandleByIndex,
                    "get_util": nvmlDeviceGetUtilizationRates,
                    "get_mem": nvmlDeviceGetMemoryInfo,
                    "get_power": nvmlDeviceGetPowerUsage
                }
                self.nvml['init']()
                self.handle = self.nvml['get_handle'](self.gpu_index)
            except ImportError:
                print("pynvml not found. Please install it for NVIDIA GPU monitoring.")
                self.gpu_vendor = 'unsupported'
        elif self.gpu_vendor == 'amd':
            try:
                from pyrsmi import rocml
                self.rocml = rocml
                self.rocml.smi_initialize()
            except ImportError:
                print("pyrsmi not found. Please install it for AMD GPU monitoring.")
                self.gpu_vendor = 'unsupported'

    def _monitor_thread(self):
        """The target function for the monitoring thread."""
        while self.monitoring:
            try:
                if self.gpu_vendor == 'nvidia':
                    util = self.nvml['get_util'](self.handle)
                    mem = self.nvml['get_mem'](self.handle)
                    power = self.nvml['get_power'](self.handle) / 1000.0  # Convert mW to W
                    self.metrics.append({
                        'timestamp': time.time(),
                        'utilization_percent': util.gpu,
                        'memory_used_mb': mem.used / (1024**2),
                        'power_watts': power
                    })
                elif self.gpu_vendor == 'amd':
                    power = self.rocml.smi_get_device_average_power(self.gpu_index)
                    mem_used = self.rocml.smi_get_device_memory_used(self.gpu_index)
                    util = self.rocml.smi_get_device_utilization(self.gpu_index)
                    self.metrics.append({
                        'timestamp': time.time(),
                        'utilization_percent': util,
                        'memory_used_mb': mem_used / (1024**2),
                        'power_watts': power
                    })
            except Exception as e:
                print(f"Metric logging error: {e}")
            time.sleep(self.interval)

    def start(self):
        """Starts the monitoring thread."""
        if self.gpu_vendor == 'unsupported':
            return
        self.monitoring = True
        self.metrics = []
        self.thread = threading.Thread(target=self._monitor_thread, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self.monitoring = False
        if self.thread:
            self.thread.join()

    def get_results(self):
        """Analyzes collected metrics and returns aggregated results."""
        if not self.metrics:
            return {
                'peak_memory_mb': 0,
                'avg_utilization_percent': 0,
                'avg_power_watts': 0
            }
        
        return {
            'peak_memory_mb': max(m['memory_used_mb'] for m in self.metrics),
            'avg_utilization_percent': statistics.mean(m['utilization_percent'] for m in self.metrics),
            'avg_power_watts': statistics.mean(m['power_watts'] for m in self.metrics)
        }, self.metrics

    def __del__(self):
        if self.gpu_vendor == 'amd' and hasattr(self, 'rocml'):
            self.rocml.smi_shutdown()