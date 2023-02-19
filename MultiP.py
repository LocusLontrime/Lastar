import multiprocessing
from multiprocessing import Pool
import time


def f(x):
    return pow(x, x, 117)


if __name__ == '__main__':
    n_cpu = multiprocessing.cpu_count()
    # multiprocessing:
    p = Pool(processes=n_cpu)
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    start = time.time_ns()
    results = p.map(f, [i for i in range(10000000)])
    # print(p.map(f, [1, 2, 3]))
    finish = time.time_ns()
    print(f"It's done!")
    print(f'time elapsed: {(finish - start) // 10 ** 6} ms')
    # serial:
    a = map(f, [i for i in range(10000000)])
    end = time.time_ns()
    print(f"It's done!")
    print(f'time elapsed: {(end - finish) // 10 ** 6} ms')

# import arcade





