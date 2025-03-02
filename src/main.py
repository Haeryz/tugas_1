import simpy
import numpy as np
import matplotlib.pyplot as plt
from simulation import ServicePlatformSimulation, AgentType
from analyzer import calculate_metrics

# Simulation parameters
# Define agent configuration instead of simple NUM_AGENTS
AGENT_CONFIGURATION = {
    AgentType.JUNIOR: 2,    # 2 junior agents
    AgentType.STANDARD: 2,  # 2 standard agents
    AgentType.SENIOR: 1     # 1 senior agent
}
# Total agents is now the sum of all agent types
NUM_AGENTS = sum(AGENT_CONFIGURATION.values())  # Still 5 agents total
MEAN_SERVICE_TIME = 10  # minutes
SERVICE_TIME_STD = 3  # standard deviation for service time
SIM_TIME = 480  # minutes (8-hour workday)
ARRIVAL_RATES = [0.2, 0.3, 0.4, 0.5, 0.6]  # customers per minute
MAX_PATIENCE = 30  # maximum patience time in minutes (None for infinite patience)

def run_simulation(arrival_rate, patience=None):
    env = simpy.Environment()
    # Pass the agent configuration dictionary instead of NUM_AGENTS
    sim = ServicePlatformSimulation(env, AGENT_CONFIGURATION, MEAN_SERVICE_TIME, SERVICE_TIME_STD)
    env.process(sim.customer_arrival_process(arrival_rate, patience))
    env.run(until=SIM_TIME)
    return calculate_metrics(sim, SIM_TIME)

def run_scenario(scenario_name, patience=None):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    results = []
    
    for rate in ARRIVAL_RATES:
        metrics = run_simulation(rate, patience)
        results.append(metrics)
        print(f"Arrival rate: {rate:.1f} customers/min")
        print(f"  Avg waiting time: {metrics['avg_waiting_time']:.2f} min")
        print(f"  Utilization: {metrics['utilization']:.2f}")
        print(f"  Completed customers: {metrics['completed_customers']}")
        if 'abandonment_rate' in metrics:
            print(f"  Abandonment rate: {metrics['abandonment_rate']:.2%}")
    
    return results

def plot_results(scenario_results, scenario_names):
    metrics_to_plot = [
        ('avg_waiting_time', 'Average Waiting Time (min)'),
        ('utilization', 'System Utilization'),
        ('abandonment_rate', 'Abandonment Rate')
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, (metric, ylabel) in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, i)
        for scenario_idx, results in enumerate(scenario_results):
            if metric in results[0]:  # Check if metric exists
                values = [res[metric] for res in results]
                plt.plot(ARRIVAL_RATES, values, marker='o', label=scenario_names[scenario_idx])
        
        plt.xlabel('Arrival Rate (customers/min)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Arrival Rate')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Queue length vs utilization scatter plot
    plt.subplot(2, 2, 4)
    for scenario_idx, results in enumerate(scenario_results):
        utils = [res['utilization'] for res in results]
        wait_times = [res['avg_waiting_time'] for res in results]
        plt.scatter(utils, wait_times, label=scenario_names[scenario_idx])
        
    plt.xlabel('System Utilization')
    plt.ylabel('Average Waiting Time (min)')
    plt.title('Waiting Time vs Utilization')
    plt.grid(True, alpha=0.3)
    plt.legend()
        
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.close()

if __name__ == "__main__":
    # Run different scenarios
    scenario_results = []
    scenario_names = []
    
    # Scenario 1: Infinite patience
    results = run_scenario("Infinite Patience", patience=None)
    scenario_results.append(results)
    scenario_names.append("Infinite Patience")
    
    # Scenario 2: Limited patience
    results = run_scenario("Limited Patience", patience=MAX_PATIENCE)
    scenario_results.append(results)
    scenario_names.append("Limited Patience")
    
    # Plot comparative results
    plot_results(scenario_results, scenario_names)
    
    print("\nSimulation complete. Results saved to 'simulation_results.png'")