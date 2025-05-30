import time

def timer(msg):
    start_time = time.time()
    format_time = lambda time: f"{int(time // 3600):02}h:{int((time % 3600) // 60):02}m:{int(time % 60):02}s"
    while True:
        elapsed_time = time.time() - start_time
        print(f'[{msg}]->Time elapsed: {format_time(elapsed_time)}', end='\r')
        time.sleep(1)