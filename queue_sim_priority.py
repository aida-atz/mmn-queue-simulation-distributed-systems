import argparse
import collections
import logging
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from random import expovariate, sample, seed, choice, randint
from discrete_event_sim_priority import Simulation, Event
class MMN(Simulation):
    def __init__(self, lambd, mu, n, d, max_t,plot_interval):
        super().__init__()
        self.running = [None] * n
        self.queues: list[collections.deque] = [collections.deque() for _ in range(n)]  # FIFO queues of the system  
        self.arrivals: dict[int, float] = {} 
        self.completions: dict[int, float] = {} 
        self.lambd = lambd
        self.n = n
        self.d = d
        self.max_t = max_t
        self.mu = mu
        self.times = []
        self.arrival_rate = lambd * n
        self.completion_rate = mu
        self.array_len = [0] * self.n
        self.schedule(expovariate(lambd), Arrival(0, 1))
        self.schedule(0, MonitoringQueueSize(plot_interval))

    def schedule_arrival(self, job_id , priority):
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id, priority), priority)

    def schedule_completion(self, job_id, queue_index,priority):
        self.schedule(expovariate(self.mu), Completion(job_id, queue_index),priority)

    def queue_len(self, i):
        return (self.running[i] is not None) + len(self.queues[i])

class MonitoringQueueSize(Event):
    def __init__(self, interval=10):
        self.interval = interval
    def process(self, sim: MMN):
        for i in range(0, sim.n):
            sim.times.append(sim.queue_len(i))
        sim.schedule(self.interval, self)

class Arrival(Event):
    def __init__(self, job_id, priority):
        self.id = job_id
        self.priority = priority
    def process(self, sim: MMN):
        sim.arrivals[self.id] = sim.t
        samples = sample(sim.queues, sim.d)
        shortest_lists = [lst for lst in samples if len(lst) == min(len(sublist) for sublist in samples)]
        selected_list = choice(shortest_lists)
        queue_index = sim.queues.index(selected_list)
        if sim.running[queue_index] is None:
            sim.running[queue_index] = self.id
            sim.schedule_completion(self.id, queue_index, self.priority)
        else:
            sim.queues[queue_index].append((self.priority, self.id)) 
        next_priority = randint(1, 5)
        sim.schedule_arrival(self.id + 1, next_priority)

class Completion(Event):
    def __init__(self, job_id, queue_index):
        self.job_id = job_id
        self.queue_index = queue_index
    def process(self, sim: MMN): 
        queue_index = self.queue_index
        assert sim.running[queue_index] == self.job_id
        sim.completions[self.job_id] = sim.t
        if sim.queues[queue_index]:
            priority, new_job = sim.queues[queue_index].popleft()
            sim.running[queue_index] = new_job
            sim.schedule_completion(new_job, queue_index, priority)
        else: 
            sim.running[queue_index] = None
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, nargs='+', default=[0.5, 0.9, 0.95, 0.99])
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--plot_interval", type=float, default=10, help="how often to collect data points for the plot")
    args = parser.parse_args()
    if args.seed:
        seed(args.seed)
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'orange', 'green', 'red']
    plt.figure(figsize=(10, 6))

    for i, lambd_value in enumerate(args.lambd):
        sim = MMN(lambd_value, args.mu, args.n, args.d, args.max_t, args.plot_interval)
        sim.run(args.max_t)
        completions = sim.completions
        print(len(sim.arrivals))
        W = (sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions)) / len(completions)
        print(f"Average time spent in the system: {W}")
        counts = [0] * 15
        queueLengths = sim.times
        for length in queueLengths:
            if length == 0:
                continue
            for t in range(min(length, 15)):
                counts[t] += 1
        fractions = [count / len(queueLengths) for count in counts]
        plt.plot(range(1, 15), fractions[1:], color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)] ,label=f'Theoretical λ = {lambd_value}')
    plt.xlabel('Queues Length')
    plt.ylabel('Fraction of Queues with at least that size')
    plt.title(f"Practical Queues Length | n: {args.n} | d: {args.d} ")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(0, 16)
    plt.xticks(range(0, 16))
    plt.show()

if __name__ == '__main__':
    main()
