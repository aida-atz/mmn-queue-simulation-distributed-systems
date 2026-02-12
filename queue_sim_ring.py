#!/usr/bin/env python3

import argparse
import collections
import logging
import matplotlib.pyplot as plt
import random
from discrete_event_sim import Simulation, Event
from workloads import weibull_generator
from ring_toplogy import RingTopology
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
    def __init__(self, lambd, mu, n, d,max_t,arrival_shape,service_shape,interval=10,isUsedExtention = True):
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
        self.ring = RingTopology(n)
        self.completion_rate = mu
        self.arrival_gen=weibull_generator(arrival_shape,1/self.arrival_rate)
        self.service_gen=weibull_generator(service_shape,1/self.completion_rate)
        self.schedule(self.arrival_gen(), Arrival(0,self.super_market()))  # schedule the first arrival
        self.interval = interval
        self.schedule(0, RecordignQueueLength())
        self.isUsedExtention:bool = isUsedExtention
    def schedule_arrival(self, job_id):
        """Schedule the arrival of a new job."""
        print(f"Generated interarrival time: {self.arrival_gen()}")
        self.schedule(self.arrival_gen(), Arrival(job_id,self.super_market()))
    def schedule_completion(self, job_id, queue_index): 
        """Schedule the completion of a job."""
        print(f"Generated service time: {self.service_gen()}")
        self.schedule(self.service_gen(), Completion(job_id,queue_index))
    def super_market(self):
        # Random starting node index
        start_index = random.randint(0, len(self.ring.nodes) - 1)
        # Traverse the ring to get `d` nodes starting from `start_index`
        selected_indices = self.ring.traverse_ring(start_index, self.d)
        # Retrieve the corresponding queues for the selected nodes
        selected_queues = [self.queues[i] for i in selected_indices]
        # Find the shortest queues
        shortest_queues = [q for q in selected_queues if len(q) == min(len(queue) for queue in selected_queues)]
        # Randomly choose one of the shortest queues
        selected_queue = random.choice(shortest_queues)
        # Return the index of the chosen queue
        return self.queues.index(selected_queue)
    def queue_len(self, i):
        """Return the length of the i-th queue.
        
        Notice that the currently running job is counted even if it is not in self.queues[i]."""

        return (self.running[i] is not None) + len(self.queues[i])
    
class RecordignQueueLength(Event):
    def process(self, sim: MMN):
        for i in range(0, sim.n):
            sim.times.append(sim.queue_len(i))
            print(f"Time {sim.t}: Queue {i} length = {sim.queue_len(i)}")        
        sim.schedule(sim.interval, self)
        
class Arrival(Event):
    """Event representing the arrival of a new job."""

    def __init__(self, job_id,incoming_queue):
        self.id = job_id
        self.incoming_queue = incoming_queue
    def process(self, sim: MMN):  # TODO: complete this method
        sim.arrivals[self.id] = sim.t  # set the arrival time of the job
        # implement the following logic:
        # if there is no running job in the queue:
            # set the incoming one
            # schedule its completion
        if bool(sim.running) and (sim.running[self.incoming_queue]) is None:
            sim.running[self.incoming_queue] = self.id
            sim.schedule_completion(self.id,self.incoming_queue)
            print(f"Job {self.id} started running in queue {self.incoming_queue}")
        # otherwise, put the job into the queue
        else : 
            sim.queues[self.incoming_queue].append(self.id)
            print(f"Job {self.id} added to queue {self.incoming_queue}, queue length: {len(sim.queues[self.incoming_queue])}")
        # schedule the arrival of the next job
        sim.schedule_arrival(self.id +1)
        # if you are looking for inspiration, check the `Completion` class below


class Completion(Event):
    """Job completion."""

    def __init__(self, job_id, queue_index):
        self.job_id = job_id  # currently unused, might be useful when extending
        self.queue_index = queue_index

    def process(self, sim: MMN):
        # print(f"Job {self.job_id} completed at time {sim.t} in queue {self.queue_index}")
        assert sim.running[self.queue_index] == self.job_id  # the job must be the one running
        sim.completions[self.job_id] = sim.t
        queue = sim.queues[self.queue_index]
        # check if the queue is not empty
        if len(queue)>0:  # queue is not empty
            # assign the first job in the queue
            sim.running[self.queue_index]=sim.queues[self.queue_index].popleft()
            sim.schedule_completion(sim.running[self.queue_index], self.queue_index)  # schedule its completion
        elif sim.isUsedExtention:
            choosen_queue = sim.super_market() 
            if choosen_queue != self.queue_index and len(sim.queues[choosen_queue])>0:
                sim.running[self.queue_index] = sim.queues[choosen_queue].popleft()
                sim.schedule_completion(sim.running[self.queue_index],self.queue_index)
            else:
                sim.running[self.queue_index] = None
        else:
            sim.running[self.queue_index] = None  # no job is running on the queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float, nargs='+', default=[0.5,0.9,0.95,0.99])
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=100)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--arrival-shape', type=float, default=1, help="Shape parameter for arrival Weibull distribution")
    parser.add_argument('--service-shape', type=float, default=1, help="Shape parameter for service Weibull distribution")
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed",default=3)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    # params = [getattr(args, column) for column in CSV_COLUMNS[:-1]]
    # corresponds to params = [args.lambd, args.mu, args.max_t, args.n, args.d]

    # if any(x <= 0 for x in params):
    #     logging.error("lambd, mu, max-t, n and d must all be positive")
    #     exit(1)

    if args.seed:
        random.seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        # output info on stderr
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')
    colors = ['blue', 'orange', 'green','red']
    line_styles = ['-', '--', '-.', ':']
    plt.figure(figsize=(10, 6))
    for i, lambd_value in enumerate(args.lambd):
        sim = MMN(lambd_value, args.mu, args.n, args.d, args.max_t, args.arrival_shape, args.service_shape)
        sim.run(args.max_t)
    # if args.lambd >= args.mu:
    #     logging.warning("The system is unstable: lambda >= mu")

    # sim = MMN(args.lambd, args.mu, args.n, args.d)

        completions = sim.completions
        # print(len(sim.arrivals))
        W = ((sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions))
         / len(completions))
    # if args.csv is not None:
    #     with open(args.csv, 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(params + [W])
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
    plt.title(f"Theoretical Queues Length | n: {args.n} | d: {args.d} ")
    plt.legend()
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.ylim(0, 1)
    plt.xlim(0, 14.5)
    plt.show()



if __name__ == '__main__':
    main()
