MMN Queue Simulation – Distributed Computing

Discrete Event Simulation (DES) of M/M/N queueing systems with multiple load balancing strategies and service time distributions.

This project was developed for the Distributed Computing course under the supervision of Prof. Matteo Dell'Amico 

MMN

.

📌 Project Overview

This project investigates the behavior of distributed queueing systems through simulation.

Starting from a basic M/M/1 FIFO model, the simulator is extended to:

M/M/N (multiple servers)

Supermarket Model (Power of d Choices)

Weibull service time distribution

Priority scheduling

Ring topology for structured node selection

Optimized ring topology with finger tables

The goal is to analyze how different scheduling and load balancing strategies affect system performance under varying traffic loads.

⚙️ Core Concepts
Queueing Model

λ (lambda) – Arrival rate

μ (mu) – Service rate

n – Number of servers

max_t – Maximum simulation time

d – Number of sampled queues (Power of d Choices)

When:

d = 1 → random queue selection

d > 1 → Supermarket model (choose shortest among sampled queues)

As described in the report 

MMN

, increasing d significantly improves load balancing.

🧠 Implemented Models
1️⃣ M/M/N Queue

Extension of M/M/1 to multiple servers by:

Replacing a single queue with an array of queues

Supporting parallel job execution

Updating arrival and completion logic accordingly 

MMN

2️⃣ Supermarket Model (Power of d Choices)

Instead of assigning jobs randomly:

Sample d queues

Select the shortest one

Key Finding:

Increasing d drastically reduces long queues

Most improvement observed at d = 5

Beyond that, marginal gains 

MMN

3️⃣ Theoretical vs Practical Validation

The simulator compares:

Theoretical queue length distribution

Practical simulation results

Plots confirm strong agreement between theory and simulation 

MMN

.

4️⃣ Weibull Service Distribution

Service times were extended beyond exponential distribution using Weibull distribution:

Shape = 1 → Equivalent to exponential (memoryless)

Shape > 1 → More concentrated distribution

Shape < 1 → Heavy-tailed distribution

Key Insight:

Heavy-tailed distributions (shape < 1) introduce long jobs that degrade performance significantly 

MMN

.

5️⃣ Priority Scheduling

High-priority jobs are placed at the top of the heap queue.

Observations:

Most effective at d = 5

Little impact at low d (1,2)

Redundant at very high d (10) 

MMN

6️⃣ Ring Topology for Node Selection

Instead of randomly sampling nodes:

Nodes are organized in a ring topology

Selection follows ring-based structure

Further optimized using finger tables

This introduces a more structured distributed selection mechanism.

📊 Main Experimental Findings

Increasing d improves load balancing

Optimal balance achieved at d = 5

Heavy-tailed workloads significantly increase congestion

Priority scheduling is beneficial only when queue selection is moderately optimized

Structured node selection (ring topology) affects performance characteristics

🛠 How to Run

Example:

python queue_sim.py


For Weibull distribution:

python queue_sim_weibull.py


For priority scheduling:

python queue_sim_priority.py


(Adjust parameters inside the files as needed.)

📄 Full Report

The complete assignment report is available here:

📎 report/MMN_Report.pdf

👩‍💻 Authors

Seyede Aida Atarzadeh Hosseini

Delaram Doroudgarian

🎓 Course Context

Distributed Computing
University of Genoa
DIBRIS Department
