import datetime
import multiprocessing as mp
import os
import time


def withpid(s):
    print(f"{os.getpid()} - {datetime.datetime.now()}: {s}")


def pippo(args):
    i, wait_time = args
    withpid(f"sleep {wait_time} seconds...")
    time.sleep(wait_time)
    withpid(f"pippo -> {i}")
    return i


if __name__ == "__main__":
    print(datetime.datetime.now())
    with mp.Pool(processes=4) as pool:
        res = pool.map(pippo, zip(*[list(range(5))] * 2))
    print(res)
    print(datetime.datetime.now())

    # ps = []
    # ps.append(mp.Process(target=pippo, args=(0,)))
    # ps[0].start()
    # time.sleep(2)
    # for i in range(1, 5):
    #     p = mp.Process(target=pippo, args=(i,))
    #     ps.append(p)
    #     p.start()
    # for p in ps:
    #     x = p.join()
