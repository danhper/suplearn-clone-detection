import os


TF_GPU_MAX_MEMORY_USAGE = float(os.environ.get("TF_GPU_MAX_MEMORY_USAGE", "0.9"))
