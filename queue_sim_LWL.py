import argparse
import collections
import logging
import matplotlib.pyplot as plt
from random import expovariate, sample, seed, choice, uniform
from discrete_event_sim import Simulation, Event
class MMN(Simulation):
    def __init__(self, lambd, mu, n, d, max_t,plot_interval, LWL_extention, hybrid_mode):
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
        self.schedule(expovariate(self.arrival_rate), Arrival(0))
        self.schedule(0, MonitoringQueueSize(plot_interval))
        self.LWL_extention : bool = LWL_extention
        self.hybrid_mode : bool = hybrid_mode
    def schedule_arrival(self, job_id):
        self.schedule(expovariate(self.arrival_rate), Arrival(job_id))

    def schedule_completion(self,processing_time, job_id, queue_index):
        self.schedule(processing_time, Completion(job_id, queue_index))

    def queue_len(self, i):
        return (self.running[i] is not None) + len(self.queues[i])
    
    def work_left(self, queue):
        """Compute the total remaining work in a queue based on actual job durations."""
        queue_index = self.queues.index(queue)
        running_time = self.running[queue_index][1] if self.running[queue_index] is not None else 0
        queue_time = sum(job[1] for job in queue)
        work_left_time = running_time + queue_time
        return work_left_time
    
    
class MonitoringQueueSize(Event):
    def __init__(self, interval=10):
        self.interval = interval
    def process(self, sim: MMN):
        for i in range(0, sim.n):
            sim.times.append(sim.queue_len(i))
        sim.schedule(self.interval, self)

class Arrival(Event):
    def __init__(self, job_id):
        self.id = job_id
    def process(self, sim: MMN):
        sim.arrivals[self.id] = sim.t
        samples = sample(sim.queues, sim.d)
        selected_list = []
        if sim.LWL_extention:
            if sim.hybrid_mode:
                work_left_list = [(q, sim.work_left(q)) for q in samples]
                min_work_queue = min(work_left_list, key=lambda x: x[1])
                long_running_threshold = 1.5 * (1 / sim.completion_rate)
                running_job_time = 0
                # running_job_time = sim.running[sim.queues.index(min_work_queue[0])][1] if sim.running[sim.queues.index(min_work_queue[0])] is not None else 0
                if sim.running[sim.queues.index(min_work_queue[0])]:
                    running_job_time = sim.running[sim.queues.index(min_work_queue[0])][1]
                else: 
                    running_job_time=0
                if running_job_time > long_running_threshold:
                        selected_list = choice([lst for lst in samples if len(lst) == min(len(sublist) for sublist in samples)])
                else:
                    selected_list = min_work_queue[0]
            else:
                selected_list = min(samples, key=lambda q: sim.work_left(q))  # Choose queue with least work left
        else:
            shortest_lists = [lst for lst in samples if len(lst) == min(len(sublist) for sublist in samples)]
            selected_list = choice(shortest_lists)
        processing_time = expovariate(sim.completion_rate)  # Assign random service time
        queue_index = sim.queues.index(selected_list)
        if sim.running[queue_index] is None:
            sim.running[queue_index] = (self.id, processing_time) 
            sim.schedule_completion(processing_time ,self.id, queue_index)
        else:
            sim.queues[queue_index].append((self.id,processing_time))
        sim.schedule_arrival(self.id + 1)

class Completion(Event):
    def __init__(self, job_id, queue_index):
        self.job_id = job_id
        self.queue_index = queue_index
    def process(self, sim: MMN): 
        queue_index = self.queue_index
        assert sim.running[queue_index][0] == self.job_id
        sim.completions[self.job_id] = sim.t
        if sim.queues[queue_index]:
            new_job,processing_time = sim.queues[queue_index].popleft()
            sim.running[queue_index] = (new_job,processing_time)
            sim.schedule_completion(processing_time,new_job, queue_index)
        else: 
            sim.running[queue_index] = None
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--lambd', type=float,  nargs='+', default=[0.5,0.6,0.77,0.8,0.85,0.9,0.95,0.99])
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--max-t', type=float, default=1_000)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--d', type=int, nargs='+', default=[1, 2, 5, 10])
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument("--seed", help="random seed")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--LWL-extention", action='store_true',default=True)
    parser.add_argument('--hybrid-mode', action='store_true', default=True)
    parser.add_argument("--plot_interval", type=float, default=10, help="how often to collect data points for the plot")

    args = parser.parse_args()
    if args.seed:
        seed(args.seed)
    if args.verbose:
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')  # output info on stdout
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'orange', 'green', 'red']
    plt.figure(figsize=(10, 6))

    for j, d_value in enumerate(args.d):
        print(f"d:{d_value}")
        W_values = [] 
        for lambd_value in args.lambd:
            sim = MMN(lambd_value, args.mu, args.n, d_value, args.max_t, args.plot_interval,args.LWL_extention,args.hybrid_mode)
            sim.run(args.max_t)
            W = (sum(sim.completions.values()) - sum(sim.arrivals[job_id] for job_id in sim.completions)) / len(sim.completions)
            print(f"Average time spent in the system: {W}")
            W_values.append(W)
        plt.plot(args.lambd, W_values, label=f'mu={args.mu}, max_t={args.max_t}, n={args.n}, d={d_value}', linestyle=line_styles[j % len(line_styles)], color=colors[j % len(colors)])
    plt.xlabel('λ')
    plt.ylabel('W')
    plt.yscale("log")  
    plt.yscale("log")  
    plt.title(f"System Performance: W vs. λ")
    plt.legend()
    plt.grid(True)
    plt.show()    
if __name__ == '__main__':
    main()
