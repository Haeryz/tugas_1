import simpy
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
from simulation import ServicePlatformSimulation, AgentType
from analyzer import calculate_metrics, create_metrics_dataframe
from data_loader import load_call_center_data, process_call_data, visualize_call_data, validate_simulation_against_data
from performance_analyzer import visualize_performance, create_performance_report, analyze_resource_utilization

# Simulation parameters
# Define agent configuration instead of simple NUM_AGENTS
AGENT_CONFIGURATION = {
    AgentType.JUNIOR: 2,    # 2 junior agents
    AgentType.STANDARD: 2,  # 2 standard agents
    AgentType.SENIOR: 1     # 1 senior agent
}
# Total agents is now the sum of all agent types
NUM_AGENTS = sum(AGENT_CONFIGURATION.values())  # Still 5 agents total

# Default parameters (will be updated from real data if available)
MEAN_SERVICE_TIME = 10  # minutes
SERVICE_TIME_STD = 3  # standard deviation for service time
SIM_TIME = 480  # minutes (8-hour workday)
ARRIVAL_RATES = [0.2, 0.3, 0.4, 0.5, 0.6]  # customers per minute
MAX_PATIENCE = 30  # maximum patience time in minutes (None for infinite patience)

# Create output directory for results
OUTPUT_DIR = "d:\\semester_6\\modeling\\tugas_1\\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data file path - update this to your dataset location
DATA_FILE = "d:\\semester_6\\modeling\\tugas_1\\data\\call_center_data.csv"

def run_simulation(arrival_rate, patience=None, service_mean=None, service_std=None):
    env = simpy.Environment()
    # Use parameters from data if provided, otherwise use defaults
    mean_service = service_mean if service_mean is not None else MEAN_SERVICE_TIME
    std_service = service_std if service_std is not None else SERVICE_TIME_STD
    
    # Pass the agent configuration dictionary instead of NUM_AGENTS
    sim = ServicePlatformSimulation(env, AGENT_CONFIGURATION, mean_service, std_service)
    env.process(sim.customer_arrival_process(arrival_rate, patience))
    env.run(until=SIM_TIME)
    return calculate_metrics(sim, SIM_TIME)

def run_scenario(scenario_name, patience=None, service_mean=None, service_std=None):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    results = []
    
    for rate in ARRIVAL_RATES:
        start_time = time.time()
        metrics = run_simulation(rate, patience, service_mean, service_std)
        elapsed = time.time() - start_time
        
        results.append(metrics)
        print(f"Arrival rate: {rate:.1f} customers/min (simulation took {elapsed:.2f}s)")
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
    plt.savefig(f'{OUTPUT_DIR}/simulation_results.png')
    plt.close()

def calibrate_model_with_data(data_metrics):
    """Update simulation parameters based on real data metrics"""
    global MEAN_SERVICE_TIME, SERVICE_TIME_STD, ARRIVAL_RATES
    
    print("\n--- Calibrating Model with Real Data ---")
    
    # Update service time parameters if available
    if 'mean_service_time' in data_metrics:
        MEAN_SERVICE_TIME = data_metrics['mean_service_time']
        print(f"Updated mean service time to: {MEAN_SERVICE_TIME:.2f} minutes")
    
    if 'std_service_time' in data_metrics:
        SERVICE_TIME_STD = data_metrics['std_service_time']
        print(f"Updated service time std dev to: {SERVICE_TIME_STD:.2f}")
    
    # Adjust arrival rates based on data - create range around observed average
    if 'avg_arrival_rate' in data_metrics:
        base_rate = data_metrics['avg_arrival_rate']
        # Create range from 60% to 140% of observed rate
        ARRIVAL_RATES = [round(base_rate * factor, 2) for factor in [0.6, 0.8, 1.0, 1.2, 1.4]]
        print(f"Updated arrival rates based on data: {ARRIVAL_RATES}")
    
    # If we have hourly pattern data, update the simulation to use it
    if 'hourly_pattern' in data_metrics:
        # Here we'll pass the pattern to the simulation class when we instantiate it
        print("Hourly call patterns loaded from data")

    return {
        'mean_service_time': MEAN_SERVICE_TIME,
        'service_time_std': SERVICE_TIME_STD,
        'arrival_rates': ARRIVAL_RATES
    }

def compare_with_real_data(simulation_results, data_metrics):
    """Compare simulation results with real data metrics"""
    print("\n--- Comparing Simulation with Real Data ---")
    
    # Find the simulation scenario with arrival rate closest to real data
    if 'avg_arrival_rate' in data_metrics:
        real_rate = data_metrics['avg_arrival_rate']
        closest_idx = min(range(len(ARRIVAL_RATES)), 
                          key=lambda i: abs(ARRIVAL_RATES[i] - real_rate))
        
        # Get simulation results for that rate
        sim_results = simulation_results[closest_idx]
        
        # Compare key metrics
        comparison = validate_simulation_against_data(sim_results, data_metrics)
        
        for key, value in comparison.items():
            if 'pct_error' in key:
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value:.4f}")
                
        return comparison
    
    return None

def run_performance_analysis(scenario_results, scenario_names):
    """Run detailed performance analysis and generate reports"""
    print("\n--- Running Advanced Performance Analysis ---")
    
    # Create detailed visualizations
    visualize_performance(scenario_results, scenario_names, ARRIVAL_RATES, OUTPUT_DIR)
    
    # Create performance report
    report_path = os.path.join(OUTPUT_DIR, "performance_report.md")
    create_performance_report(scenario_results, scenario_names, ARRIVAL_RATES, 
                             AGENT_CONFIGURATION, report_path)
    
    print(f"Performance analysis completed. Reports saved to {OUTPUT_DIR}/")
    
    # Run utilization analysis for each scenario and display key insights
    for i, scenario_results_list in enumerate(scenario_results):
        analysis = analyze_resource_utilization(scenario_results_list, ARRIVAL_RATES, AGENT_CONFIGURATION)
        
        print(f"\nKey insights for {scenario_names[i]}:")
        if 'critical_utilization' in analysis:
            print(f"- Critical utilization point: {analysis['critical_utilization']:.2f}")
        if 'optimal_arrival_rate' in analysis:
            print(f"- Optimal arrival rate: {analysis['optimal_arrival_rate']:.2f} customers/min")
        if 'recommended_arrival_rate' in analysis:
            print(f"- Recommended max arrival rate: {analysis['recommended_arrival_rate']:.2f} customers/min")

if __name__ == "__main__":
    # Set up the style for visualizations
    sns.set(style="whitegrid", context="talk")
    
    # First, try to load real call center data if available
    try:
        print(f"Attempting to load call center data from: {DATA_FILE}")
        call_data = load_call_center_data(DATA_FILE)
        data_metrics = process_call_data(call_data)
        
        print("\n--- Call Center Data Loaded Successfully ---")
        print(f"Total records: {len(call_data)}")
        print(f"Average wait time: {data_metrics.get('mean_wait_time', 'N/A'):.2f} minutes")
        print(f"Average service time: {data_metrics.get('mean_service_time', 'N/A'):.2f} minutes")
        
        # Visualize the data
        viz_file = visualize_call_data(call_data)
        print(f"Data visualizations saved to: {viz_file}")
        
        # Calibrate model using real data
        calibrated_params = calibrate_model_with_data(data_metrics)
        
        data_available = True
    except Exception as e:
        print(f"Could not load call center data: {str(e)}")
        print("Proceeding with default simulation parameters")
        data_available = False
    
    # Run scenarios with calibrated parameters if data is available
    scenario_results = []
    scenario_names = []
    
    # Scenario 1: Infinite patience
    if data_available:
        results = run_scenario("Infinite Patience (Calibrated)", 
                              patience=None,
                              service_mean=calibrated_params['mean_service_time'],
                              service_std=calibrated_params['service_time_std'])
    else:
        results = run_scenario("Infinite Patience", patience=None)
        
    scenario_results.append(results)
    scenario_names.append("Infinite Patience")
    
    # Scenario 2: Limited patience
    if data_available:
        results = run_scenario("Limited Patience (Calibrated)",
                              patience=MAX_PATIENCE,
                              service_mean=calibrated_params['mean_service_time'],
                              service_std=calibrated_params['service_time_std'])
    else:
        results = run_scenario("Limited Patience", patience=MAX_PATIENCE)
        
    scenario_results.append(results)
    scenario_names.append("Limited Patience")
    
    # Plot comparative results
    plot_results(scenario_results, scenario_names)
    
    # Run detailed performance analysis
    run_performance_analysis(scenario_results, scenario_names)
    
    # Compare with real data if available
    if data_available:
        for scenario_idx, scenario_results_list in enumerate(scenario_results):
            print(f"\nValidating {scenario_names[scenario_idx]} against real data:")
            compare_with_real_data(scenario_results_list, data_metrics)
    
    # Create metrics dataframe for easier additional analysis
    metrics_df = create_metrics_dataframe(scenario_results, scenario_names, ARRIVAL_RATES)
    metrics_df.to_csv(f'{OUTPUT_DIR}/metrics_summary.csv', index=False)
    
    print("\nSimulation complete. Results saved to output directory.")