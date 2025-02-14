import os
from collections import defaultdict
import networkx as nx
import math
import random

CPUS = 64
DEVICES = 28
Q_PER_DEVICE = 17

def parse_cpu_list(cpu_list):
    """
    Parse a cpu_list string (e.g., "6, 7, 70, 71") into a list of integers.
    """
    return [int(cpu.strip()) for cpu in cpu_list.split(",")]

def translate_files_to_mapping(base_path):
    queue_to_hosts = defaultdict(lambda : defaultdict(list))
    devices = sorted([d for d in os.listdir(base_path) if d.isdigit()])
    for device in devices:
        device_path = os.path.join(base_path, device)
        mqs_path = os.path.join(device_path, "mqs")
        mqs = sorted([mq for mq in os.listdir(mqs_path) if mq.isdigit() and mq != "0"])
        for mq in mqs:
            cpu_list_path = os.path.join(mqs_path, mq, "cpu_list")
            if not os.path.exists(cpu_list_path):
                continue
            with open(cpu_list_path, "r") as f:
                cpu_list = parse_cpu_list(f.read().strip())
            queue_to_hosts[str(device)][str(mq)] = cpu_list
    return queue_to_hosts

def convert_flow_to_res(flow_dict):
    flow_cpus = defaultdict(list)
    for u in flow_dict:
	    if "_" in str(u):
		    for v in flow_dict[u]:
			    if flow_dict[u][v] == 1:
				    flow_cpus[u.split("_")[0]].append(int(v))
    return  [','.join(map(str, x[1])) for x in (sorted(flow_cpus.items())) ]

def solve_for_min_max_jobs(queue_to_hosts, minJobsOnCpu, maxJobsOnCpu, num_queues):
    #dv id incoming-outgoing flow
    G = nx.DiGraph()
    G.add_node("end", required_dv = 1 * num_queues, curr_dv = 0)

    for device, q_to_host in queue_to_hosts.items():
        for qq, cpus in q_to_host.items():
            q = str(device) + "_" + str(qq)
            G.add_node(q, required_dv = -1, curr_dv = 0)
            for cpu in cpus:
                if not G.has_node(cpu):
                    G.add_node(cpu, required_dv = 0, curr_dv = 0)
                    G.add_edge(cpu, "end", capacity=maxJobsOnCpu)
                    #here is the magic: run the minimal requird flow 
                    G.nodes[cpu]['curr_dv'] -= minJobsOnCpu
                    G.nodes["end"]['curr_dv'] += minJobsOnCpu
                    G.edges[(cpu, "end")]["capacity"] -= minJobsOnCpu
                    #end of magic 
                G.add_edge(q, cpu, capacity=1)
    G.add_node("source")
    G.add_node("sink")
    needed_flow = 0
    for node, attr in G.nodes.items():
        if node not in ["source", "sink"]:
            new_dv = attr["required_dv"] - attr["curr_dv"]
            if new_dv > 0:
                G.add_edge(node, "sink", capacity = new_dv)
                needed_flow += new_dv
            elif new_dv < 0:
                G.add_edge("source", node, capacity = (-1) * new_dv)
    flow_value, flow_dict = nx.maximum_flow(G, 'source', 'sink')
    return convert_flow_to_res(flow_dict) if flow_value == needed_flow else []



def find_best_solution_max_flow(queue_to_hosts):
    all_cpus, num_queues = set(), 0
    for device, q_to_host in queue_to_hosts.items():
        for q, cpus in q_to_host.items():
            num_queues += 1
            all_cpus.update(cpus)
    avg = num_queues // len(all_cpus)
    highBoundAvg, lowBoundAvg = math.ceil(avg), int(avg)

    for diff in range(num_queues + 1):
        maxJobsOnCpu = max(diff, highBoundAvg)
        minJobsOnCpu = maxJobsOnCpu - diff
        while maxJobsOnCpu <= num_queues and minJobsOnCpu <= lowBoundAvg:
            res = solve_for_min_max_jobs(queue_to_hosts, minJobsOnCpu, 
                    maxJobsOnCpu, num_queues)
            if len(res) > 0:
                return res
            maxJobsOnCpu += 1
            minJobsOnCpu += 1

def find_best_solution_greedy(queue_to_hosts):
    cpus = []  # Array to store CPUs for each device
    global_cpu_usage = defaultdict(int)  # Track global usage of each CPU for fairness
    for device, q_to_host in sorted(queue_to_hosts.items()):
        device_cpus = set()  # Track unique CPUs for this device
        for qq, cpu_list in q_to_host.items():
            available_cpus = [cpu for cpu in cpu_list if cpu not in device_cpus]
            if available_cpus:
                # Sort available CPUs by their global usage count for fairness
                selected_cpu = min(available_cpus, key=lambda cpu: global_cpu_usage[cpu])
                device_cpus.add(selected_cpu)
                global_cpu_usage[selected_cpu] += 1
        cpus.append(list(device_cpus))
    return [','.join([str(cpu) for cpu in device]) for device in cpus]
    
  
    


def solveFlow(base_path="/tmp/mapping"):
    return find_best_solution_max_flow(translate_files_to_mapping(base_path))  
    



def getBackMapping(queue_to_hosts):
    back_mapping = defaultdict(lambda : defaultdict(str))
    for device, q_to_host in queue_to_hosts.items():
        for qq, cpu_list in q_to_host.items():
            for cpu in cpu_list:
                back_mapping[device][cpu] = qq
    return back_mapping

def validate_solution(queue_to_hosts, sol):
    back_mapping = getBackMapping(queue_to_hosts)
    flat_forward = sorted(queue_to_hosts.items())
    flat_back = sorted(back_mapping.items())


    cpus_freq = {str(cpu):0 for cpu in flat_back[0][1]}
    

    for indx in range(len(sol)):
        assigned_cpus_str = sol[indx].split(",")
        num_queues = len(flat_forward[indx][1])
        if len(assigned_cpus_str) != num_queues:
            print ("AAAAAAA")
            return False, -1
        cpus_to_queues = flat_back[indx][1]
        used_queues = set()
        for cpu in assigned_cpus_str:
            used_queues.add(cpus_to_queues[cpu])
            cpus_freq[cpu] += 1
        if len(used_queues) != num_queues:
            print ("BBBBB", num_queues, len(used_queues))
            return False, -1

    return True, max(cpus_freq.values()) - min(cpus_freq.values()), cpus_freq
            

#######################THIS FUNCTION CREATES CONF WITH EXTREMELY BAD OPTIMAL SOLUTION##############################################
def destroy_conf_to_create_extremely_bad_optimal_solution(conf):
    for device in conf:
        conf[str(device)][Q_PER_DEVICE].append(str(CPUS))
#####################################################################
    


#######################THIS FUNCTION CREATES RANDOM CONF WITH CONST AMOUNT OF QUEUES PER CONTROLLER##############################################
def random_config():
    #{device->{q->cpus}}
    conf = defaultdict(lambda: defaultdict(list))
    cpus = [cpu for cpu in range(CPUS)]
    for device in range(DEVICES):
        delimiters = sorted(random.sample(range(CPUS - 1), Q_PER_DEVICE - 1))
        delimiters.append(CPUS - 1)
        random.shuffle(cpus)
        del_indx = 0
        for indx in range(len(cpus)):
            if indx > delimiters[del_indx]:
                del_indx += 1
            conf[str(device)][str(del_indx)].append(str(cpus[indx]))
    return conf

#####################################################################

#######################THOSE 2 FUNCTIONS CREATE A VERY BAD CONF FOR GREEDY##############################################

def arrange_badly(conf, curr_queue, curr_device, cpus):
    if len(cpus) % 4 != 0:
        raise(Exception("WTF"))
    quadr_indx = 0
    while quadr_indx < len(cpus):
        c1, c2, c3, c4 = cpus[quadr_indx], cpus[quadr_indx + 1], cpus[quadr_indx + 2], cpus[quadr_indx + 3]
        conf[str(curr_device)][str(curr_queue)] = [str(c1), str(c2)]
        conf[str(curr_device)][str(curr_queue + 1)] = [str(c3), str(c4)]
        conf[str(curr_device + 1)][str(curr_queue)] = [str(c1), str(c3)]
        conf[str(curr_device + 1)][str(curr_queue + 1)] = [str(c2), str(c4)]
        curr_queue += 2
        quadr_indx += 4
    return curr_queue


    
def create_bad_example_for_greedy():
    good, bad, ugly = [cpu for cpu in range(1024)], [], []
    conf = defaultdict(lambda: defaultdict(list))
    curr_device = 0
    while len(good) > 1:
        curr_queue = 0
        curr_queue = arrange_badly(conf, curr_queue, curr_device, good)
        curr_queue = arrange_badly(conf, curr_queue, curr_device, bad)
        curr_queue = arrange_badly(conf, curr_queue, curr_device, ugly)
        greedy_sol = find_best_solution_greedy(conf)
        valid_greedy, res_greedy, cpus_freq = validate_solution(conf, greedy_sol)
        curr_device += 2
        good, bad, ugly = [], [], []
        for cpu in cpus_freq:
            if cpus_freq[cpu] == curr_device:
                good.append(cpu)                
            elif cpus_freq[cpu] == 0:
                bad.append(cpu)
            else:
                ugly.append(cpu)
    return conf

#####################################################################
if __name__ == "__main__":
    highest_solutions_diff = 0
    worst_conf = 0
    worst_greedy = 0
    worst_flow = 0
    


    for i in range(100000):
        queue_to_hosts = random_config()
        flow_sol = find_best_solution_max_flow(queue_to_hosts)
        greedy_sol = find_best_solution_greedy(queue_to_hosts)
        valid_flow, res_flow, cpus_freq = validate_solution(queue_to_hosts, flow_sol)
        valid_greedy, res_greedy, cpus_freq = validate_solution(queue_to_hosts, greedy_sol)
        if not valid_flow:
            print("flow solution is not valid")
            break
        if not valid_greedy:
            print("greedy solution is not valid")
            break
        if res_flow > res_greedy:
            print ("WTF")
            break

        if res_greedy - res_flow > highest_solutions_diff:
            highest_solutions_diff = res_greedy - res_flow
            worst_conf = queue_to_hosts
        worst_greedy = max(worst_greedy, res_greedy)
        worst_flow = max(worst_flow, res_flow)
        print(f"highest diff till now: {highest_solutions_diff} worst greedy till now:{worst_greedy} worst flow till now: {worst_flow}  curr res greedy: {res_greedy} curr res flow: {res_flow}.")
        if(i % 500 == 0):
            print(worst_conf)
    print (best_conf)











