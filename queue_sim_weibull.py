#!/usr/bin/env python3

import argparse
import collections
import logging
import matplotlib.pyplot as plt
from random import sample, seed, choice
from discrete_event_sim import Simulation, Event
from workloads import weibull_generator
# One possible modification is to use a different distribution for job sizes or and/or interarrival times.
# Weibull distributions (https://en.wikipedia.org/wiki/Weibull_distribution) are a generalization of the
# exponential distribution, and can be used to see what happens when values are more uniform (shape > 1,
# approaching a "bell curve") or less (shape < 1, "heavy tailed" case when most of the work is concentrated
# on few jobs).

# To use Weibull variates, for a given set of parameter do something like
# from workloads import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


# columns saved in the CSV file
# CSV_COLUMNS = ['lambd', 'mu', 'max_t', 'n', 'd', 'w']

class MMN(Simulation):
    """Simulation of a system with n servers and n queues. --> m/m/n

    The system has n servers with one queue each. Jobs arrive at rate lambd and are served at rate mu.
    When a job arrives, according to the supermarket model, it chooses d queues at random and joins
    the shortest one.
    """
# lambd --> arrive rate
# mu --> serve rate
# d --> the random number of queues
# n --> the number of servers 
    def __init__(self, lambd, mu, n, d,max_t,weibull_shape,plot_interval):
        super().__init__()
        self.running:list[int] = [None] * n  # if not None, the id of the running job (per queue)
        self.queues: list[collections.deque] = [collections.deque() for _ in range(n)]  # FIFO queues of the system
        # NOTE: we don't keep the running jobs in self.queues
        self.arrivals: dict[int, float] = {}  # dictionary mapping job id to arrival time
        self.completions: dict[int, float] = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.d = d
        self.max_t=max_t # maximum simulation time: all events (arrivals,completions, queue recording) are processed up to this time.
        self.mu = mu
        self.times=[] # It stores the lengths of all queues at regular intervals throughout the simulation
        self.arrival_rate = lambd * n  # frequency of new jobs is proportional to the number of queues
        self.completion_rate = mu
        self.arrival_gen=weibull_generator(weibull_shape,1/self.arrival_rate)
        self.service_gen=weibull_generator(weibull_shape,1/self.completion_rate)
        self.schedule(self.arrival_gen(), Arrival(0))  # schedule the first arrival
        self.schedule(0, MonitoringQueueSize(plot_interval))

    def schedule_arrival(self, job_id):
        self.schedule(self.arrival_gen(), Arrival(job_id))

    def schedule_completion(self, job_id, queue_index): 
        self.schedule(self.service_gen(), Completion(job_id,queue_index))

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
    """Event representing the arrival of a new job."""

    def __init__(self, job_id):
        self.id = job_id
    def process(self, sim: MMN):
        sim.arrivals[self.id] = sim.t
        samples = sample(sim.queues, sim.d)
        shortest_lists = [lst for lst in samples if len(lst) == min(len(sublist) for sublist in samples)]
        selected_list = choice(shortest_lists)  
        queue_index = sim.queues.index(selected_list)      
        if sim.running[queue_index] is None:
            sim.running[queue_index] = self.id
            sim.schedule_completion(self.id, queue_index)
        else:
            sim.queues[queue_index].append(self.id)
        sim.schedule_arrival(self.id + 1)

class Completion(Event):
    def __init__(self, job_id, queue_index):
        self.job_id = job_id
        self.queue_index = queue_index
    def process(self, sim: MMN): 
        queue_index = self.queue_index
        assert sim.running[queue_index] == self.job_id
        sim.completions[self.job_id] = sim.t
        if sim.queues[queue_index]:
            new_job = sim.queues[queue_index].popleft()
            sim.running[queue_index] = new_job
            sim.schedule_completion(new_job, queue_index)
        else: 
            sim.running[queue_index] = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, nargs='+', default=[0.5,0.9,0.95,0.99])
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--weibull-shape', type=float, default= 0.4, help="Shape parameter for Weibull distribution")
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed",default=3)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--plot-interval", type=float, default=10, help="how often to collect data points for the plot")
    args = parser.parse_args()
    if args.seed:
        seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        # output info on stderr
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')
    colors = ['blue', 'orange', 'green','red']
    line_styles = ['-', '--', '-.', ':']
    plt.figure(figsize=(10, 6))
    for i, lambd_value in enumerate(args.lambd):
        sim = MMN(lambd_value, args.mu, args.n, args.d, args.max_t, args.weibull_shape, args.plot_interval)
        sim.run(args.max_t)
        completions = sim.completions
        W = ((sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions))
         / len(completions))
        print(f"Average time spent in the system: {W}")
        counts = [0] * 15
        queueLengths = sim.times
        for length in queueLengths:
            if length == 0:
                continue
            for t in range(min(length, 15)):
                counts[t] += 1
        fractions = [count / len(queueLengths) for count in counts]
        plt.plot(range(1, 15), fractions[1:], color=colors[i % len(colors)],linestyle=line_styles[i % len(line_styles)] ,label=f'Theoretical λ = {lambd_value}')
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
