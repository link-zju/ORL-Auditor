import GPUtil
import psutil

def select_gpu():
    gpu_id=-1
    cpu_occupy_ratio = psutil.cpu_percent(interval=1)

    try:
        gpu_id = GPUtil.getFirstAvailable(order = 'random', maxLoad=0.8, maxMemory=0.6, attempts=1, verbose=False)
    except:
        gpu_id=-1

    if cpu_occupy_ratio < 80 and gpu_id != -1:
        print(gpu_id[0])
    else:
        print("-1")

if __name__ == "__main__":
    select_gpu()

        