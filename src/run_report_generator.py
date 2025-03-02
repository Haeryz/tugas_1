import pandas as pd
import os
import sys
from analyzer import create_metrics_dataframe
from simulation import ServicePlatformSimulation, AgentType
from main import run_simulation, ARRIVAL_RATES
from report_generator import generate_all_reports

def save_simulation_results(scenarios, arrival_rates, output_dir):
    """Run simulations and save metrics for reporting"""
    scenario_results = []
    scenario_names = []
    
    # Define scenarios to run
    simulation_scenarios = {
        "Infinite Patience": {"patience": None},
        "Limited Patience": {"patience": 30},  # 30 minutes patience
    }
    
    # Run each scenario
    for scenario_name, params in simulation_scenarios.items():
        print(f"\nRunning scenario: {scenario_name}")
        scenario_metrics = []
        
        for rate in arrival_rates:
            print(f"  Arrival rate: {rate:.1f}")
            metrics = run_simulation(rate, patience=params["patience"])
            scenario_metrics.append(metrics)
        
        scenario_results.append(scenario_metrics)
        scenario_names.append(scenario_name)
    
    # Convert to DataFrame
    metrics_df = create_metrics_dataframe(scenario_results, scenario_names, arrival_rates)
    
    # Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics_file = f"{output_dir}/simulation_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    return metrics_df

if __name__ == "__main__":
    # Define paths
    output_dir = '../output'
    metrics_file = f"{output_dir}/simulation_metrics.csv"
    
    # Check if metrics file exists, run simulations if not
    if not os.path.exists(metrics_file):
        print("No metrics file found. Running simulations...")
        metrics_df = save_simulation_results(
            ["Infinite Patience", "Limited Patience"],
            ARRIVAL_RATES,
            output_dir
        )
    else:
        print(f"Loading metrics from {metrics_file}")
        metrics_df = pd.read_csv(metrics_file)
    
    # Generate report elements
    generate_all_reports(metrics_df)
    
    print("Report generation complete!")
