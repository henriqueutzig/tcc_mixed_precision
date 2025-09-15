import csv
import os
import threading
import time
from datetime import datetime

try:
    import pynvml  # NVIDIA
except ImportError:
    pynvml = None

try:
    import pyrsmi  # AMD
except ImportError:
    pyrsmi = None


class MetricsLogger:
    def __init__(self, gpu_vendor, framework, model, precision, batch_size, log_interval=5):
        self.gpu_vendor = gpu_vendor
        self.framework = framework
        self.model = model
        self.precision = precision
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.running = False
        self.thread = None

        # Output file
        os.makedirs("results/metrics", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename = f"results/metrics/{gpu_vendor}_{framework}_{model}_{precision}_{timestamp}.csv"

        # Initialize CSV
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "gpu_vendor",
                "framework",
                "model",
                "precision",
                "batch_size",
                "gpu_util",
                "memory_used",
                "power_watts"
            ])

        # Init GPU API
        if gpu_vendor == "nvidia" and pynvml:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        elif gpu_vendor == "amd" and pyrsmi:
            # AMD initialization happens on-demand
            pass

    def _log_metrics(self):
        while self.running:
            ts = datetime.now().isoformat()
            util, mem, power = self._get_gpu_metrics()

            with open(self.filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts,
                    self.gpu_vendor,
                    self.framework,
                    self.model,
                    self.precision,
                    self.batch_size,
                    util,
                    mem,
                    power
                ])

            time.sleep(self.log_interval)

    def _get_gpu_metrics(self):
        if self.gpu_vendor == "nvidia" and pynvml:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used // (1024 * 1024)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            return util, mem, power

        elif self.gpu_vendor == "amd" and pyrsmi:
            dev = pyrsmi.rsmi_dev_id_get(0)
            util = pyrsmi.rsmi_dev_busy_percent_get(dev)[1]
            mem = pyrsmi.rsmi_dev_memory_usage_get(dev, pyrsmi.RSMI_MEM_TYPE_VRAM)[1] // (1024 * 1024)
            power = pyrsmi.rsmi_dev_power_ave_get(dev, 0)[1] / 1000.0
            return util, mem, power

        return 0, 0, 0

    def start_logging(self):
        self.running = True
        self.thread = threading.Thread(target=self._log_metrics, daemon=True)
        self.thread.start()

    def stop_logging(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def finalize(self, total_time):
        print(f"📊 Metrics saved to {self.filename}")
        print(f"⏱️ Total training time: {total_time:.2f} seconds")
