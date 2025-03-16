import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def perform_sensitivity_analysis(sim_func, base_params, param_to_vary, ranges, metrics_to_track):
    """
    Perform sensitivity analysis by varying one parameter and tracking metrics.
    
    Args:
        sim_func: Function to run simulation (should accept parameters)
        base_params: Dictionary of base parameters
        param_to_vary: The parameter to vary
        ranges: List of values to use for the parameter
        metrics_to_track: List of metrics to track
    
    Returns:
        DataFrame with results
    """
    results = []
    
    for value in ranges:
        # Update the parameter
        params = base_params.copy()
        params[param_to_vary] = value
        
        # Run simulation with updated parameters
        metrics = sim_func(**params)
        
        # Store results
        row = {param_to_vary: value}
        for metric in metrics_to_track:
            row[metric] = metrics[metric]
        
        results.append(row)
    
    return pd.DataFrame(results)

def analyze_resource_utilization(scenario_results, arrival_rates, agent_config):
    """
    Analyze resource utilization and efficiency.
    
    Args:
        scenario_results: List of simulation results
        arrival_rates: List of arrival rates used
        agent_config: Dictionary of agent configuration
    
    Returns:
        Dictionary of analysis results
    """
    analysis = {}
    
    # Calculate total agents
    total_agents = sum(agent_config.values())
    
    # Calculate critical utilization point (where waiting time starts to increase rapidly)
    utils = [res['utilization'] for res in scenario_results]
    wait_times = [res['avg_waiting_time'] for res in scenario_results]
    
    # Find elbow point using simple heuristic
    wait_time_diffs = [wait_times[i+1] - wait_times[i] for i in range(len(wait_times)-1)]
    if len(wait_time_diffs) > 1:
        # Find where differences start increasing significantly
        avg_diff = np.mean(wait_time_diffs)
        critical_idx = next((i for i, diff in enumerate(wait_time_diffs) if diff > 2 * avg_diff), len(utils) - 1)
        analysis['critical_utilization'] = utils[critical_idx]
        analysis['critical_arrival_rate'] = arrival_rates[critical_idx]
    else:
        analysis['critical_utilization'] = max(utils)
        analysis['critical_arrival_rate'] = arrival_rates[-1]
    
    # Calculate efficiency metrics
    if len(scenario_results) > 0:
        # Find optimal point balancing waiting time and utilization
        scores = [(res['utilization'] * 0.7) - (res['avg_waiting_time'] / 10 * 0.3) for res in scenario_results]
        optimal_idx = scores.index(max(scores))
        
        analysis['optimal_arrival_rate'] = arrival_rates[optimal_idx]
        analysis['optimal_utilization'] = utils[optimal_idx]
        analysis['optimal_waiting_time'] = wait_times[optimal_idx]
        
        # Calculate throughput per agent
        throughputs = [res['throughput'] for res in scenario_results]
        analysis['max_throughput_per_agent'] = max(throughputs) / total_agents
        
        # Calculate revenue potential (assuming each customer generates revenue)
        completed = [res['completed_customers'] for res in scenario_results]
        analysis['revenue_potential'] = max(completed)
        
        # Calculate ideal staffing level based on Erlang C
        target_wait = 2  # target waiting time in minutes
        target_util = 0.8  # target utilization
        
        # Simple estimation based on finding closest results to targets
        combined_scores = [(abs(res['avg_waiting_time'] - target_wait) / target_wait * 0.5 + 
                          abs(res['utilization'] - target_util) / target_util * 0.5) 
                          for res in scenario_results]
        
        best_idx = combined_scores.index(min(combined_scores))
        analysis['recommended_arrival_rate'] = arrival_rates[best_idx]
        
    return analysis

def visualize_performance(scenario_results, scenario_names, arrival_rates, output_dir='.'):
    """
    Create advanced visualizations for performance analysis.
    
    Args:
        scenario_results: List of lists of scenario results
        scenario_names: Names of the scenarios
        arrival_rates: List of arrival rates
        output_dir: Directory to save output files
    """
    # Set up the style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Create a DataFrame for easier plotting
    data = []
    for scenario_idx, results in enumerate(scenario_results):
        scenario = scenario_names[scenario_idx]
        for i, result in enumerate(results):
            row = {'Scenario': scenario, 'Arrival Rate': arrival_rates[i]}
            for key, value in result.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[key] = value
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # 1. Waiting Time Analysis
    plt.figure(figsize=(12, 8))
    
    # Main waiting time plot
    plt.subplot(2, 2, 1)
    sns.lineplot(data=df, x='Arrival Rate', y='avg_waiting_time', hue='Scenario', marker='o')
    plt.title('Average Waiting Time vs Arrival Rate')
    plt.ylabel('Average Waiting Time (minutes)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Waiting time distribution
    plt.subplot(2, 2, 2)
    if 'p95_waiting_time' in df.columns and 'median_waiting_time' in df.columns:
        metrics = ['median_waiting_time', 'avg_waiting_time', 'p95_waiting_time']
        scenario = scenario_names[0]  # Use first scenario for this chart
        scenario_df = df[df['Scenario'] == scenario]
        
        for metric in metrics:
            plt.plot(scenario_df['Arrival Rate'], scenario_df[metric], 
                     label=metric.replace('_', ' ').title(), marker='o')
        
        plt.title(f'Waiting Time Metrics ({scenario})')
        plt.ylabel('Time (minutes)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    # 2. System Utilization
    plt.subplot(2, 2, 3)
    sns.lineplot(data=df, x='Arrival Rate', y='utilization', hue='Scenario', marker='o')
    plt.title('System Utilization vs Arrival Rate')
    plt.ylabel('Utilization')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Wait Time vs Utilization
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='utilization', y='avg_waiting_time', hue='Scenario', size='Arrival Rate', sizes=(50, 200))
    plt.title('Waiting Time vs Utilization')
    plt.xlabel('Utilization')
    plt.ylabel('Average Waiting Time (minutes)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/waiting_time_analysis.png', dpi=300)
    plt.close()
    
    # 4. Customer Experience Metrics
    if 'abandonment_rate' in df.columns or 'avg_customer_satisfaction' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Abandonment Rate
        if 'abandonment_rate' in df.columns:
            plt.subplot(2, 2, 1)
            sns.lineplot(data=df, x='Arrival Rate', y='abandonment_rate', hue='Scenario', marker='o')
            plt.title('Customer Abandonment Rate')
            plt.ylabel('Abandonment Rate')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        # Customer Satisfaction
        if 'avg_customer_satisfaction' in df.columns:
            plt.subplot(2, 2, 2)
            sns.lineplot(data=df, x='Arrival Rate', y='avg_customer_satisfaction', hue='Scenario', marker='o')
            plt.title('Average Customer Satisfaction')
            plt.ylabel('Satisfaction Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            
        # Completed Customers
        plt.subplot(2, 2, 3)
        sns.lineplot(data=df, x='Arrival Rate', y='completed_customers', hue='Scenario', marker='o')
        plt.title('Completed Customers')
        plt.ylabel('Number of Customers')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Throughput
        if 'throughput' in df.columns:
            plt.subplot(2, 2, 4)
            sns.lineplot(data=df, x='Arrival Rate', y='throughput', hue='Scenario', marker='o')
            plt.title('System Throughput')
            plt.ylabel('Customers per Minute')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/customer_experience.png', dpi=300)
        plt.close()
    
    # 5. Create heatmap of relationship between metrics
    plt.figure(figsize=(10, 8))
    metrics_to_correlate = ['utilization', 'avg_waiting_time', 'throughput', 
                           'completed_customers', 'abandonment_rate']
    
    # Filter only numeric columns that exist in the data
    available_metrics = [m for m in metrics_to_correlate if m in df.columns]
    
    if available_metrics:
        corr_df = df[available_metrics].corr()
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Performance Metrics')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_correlation.png', dpi=300)
        plt.close()

def create_performance_report(scenario_results, scenario_names, arrival_rates, agent_config, output_file='performance_report.md'):
    """
    Generate a detailed performance analysis report in Markdown format.
    
    Args:
        scenario_results: List of lists of scenario results
        scenario_names: Names of the scenarios
        arrival_rates: List of arrival rates
        agent_config: Dictionary of agent configuration
        output_file: File to write the report to
    """
    # Perform analysis
    analysis_results = []
    for scenario_idx, results in enumerate(scenario_results):
        analysis = analyze_resource_utilization(results, arrival_rates, agent_config)
        analysis['scenario'] = scenario_names[scenario_idx]
        analysis_results.append(analysis)
    
    # Create report content
    report = ["# Call Center Performance Analysis Report\n\n"]
    
    # System configuration section
    report.append("## System Configuration\n\n")
    report.append(f"- **Total Agents**: {sum(agent_config.values())}\n")
    for agent_type, count in agent_config.items():
        report.append(f"- **{agent_type.name}**: {count}\n")
    report.append("\n")
    
    # Performance summary for each scenario
    report.append("## Performance Summary\n\n")
    
    for analysis, scenario_name in zip(analysis_results, scenario_names):
        report.append(f"### {scenario_name}\n\n")
        report.append("| Metric | Value |\n")
        report.append("| ------ | ----- |\n")
        
        if 'critical_utilization' in analysis:
            report.append(f"| Critical Utilization Point | {analysis['critical_utilization']:.2f} |\n")
        if 'critical_arrival_rate' in analysis:
            report.append(f"| Critical Arrival Rate | {analysis['critical_arrival_rate']:.2f} customers/min |\n")
        if 'optimal_utilization' in analysis:
            report.append(f"| Optimal Utilization | {analysis['optimal_utilization']:.2f} |\n")
        if 'optimal_waiting_time' in analysis:
            report.append(f"| Optimal Waiting Time | {analysis['optimal_waiting_time']:.2f} min |\n")
        if 'max_throughput_per_agent' in analysis:
            report.append(f"| Max Throughput per Agent | {analysis['max_throughput_per_agent']:.2f} customers/min |\n")
        if 'recommended_arrival_rate' in analysis:
            report.append(f"| Recommended Arrival Rate | {analysis['recommended_arrival_rate']:.2f} customers/min |\n")
            
        report.append("\n")
    
    # Detailed metrics table
    report.append("## Detailed Metrics by Arrival Rate\n\n")
    
    for scenario_idx, results in enumerate(scenario_results):
        scenario_name = scenario_names[scenario_idx]
        report.append(f"### {scenario_name}\n\n")
        
        # Create table header
        metrics_to_show = ['utilization', 'avg_waiting_time', 'median_waiting_time', 'p95_waiting_time', 
                          'throughput', 'completed_customers', 'abandonment_rate']
        
        # Check which metrics are available
        available_metrics = []
        for metric in metrics_to_show:
            if metric in results[0]:
                available_metrics.append(metric)
        
        # Table header
        header = "| Arrival Rate | " + " | ".join(m.replace('_', ' ').title() for m in available_metrics) + " |\n"
        report.append(header)
        
        # Table separator
        separator = "| --- | " + " | ".join(["---"] * len(available_metrics)) + " |\n"
        report.append(separator)
        
        # Table rows
        for i, result in enumerate(results):
            row = f"| {arrival_rates[i]:.2f} | "
            row += " | ".join(f"{result[m]:.2f}" if isinstance(result[m], float) else str(result[m]) for m in available_metrics)
            row += " |\n"
            report.append(row)
        
        report.append("\n")
    
    # Add conclusion
    report.append("## Conclusions and Recommendations\n\n")

    # Find the best scenario based on wait time and utilization balance
    best_scenario_idx = None
    best_score = -float('inf')
    
    for scenario_idx, results in enumerate(scenario_results):
        # Calculate a score based on balanced metrics
        scenario_scores = []
        for result in results:
            score = result['utilization'] * 0.6 - (result['avg_waiting_time'] / 10) * 0.4
            if 'abandonment_rate' in result:
                score -= result['abandonment_rate'] * 5  # Penalize abandonment heavily
            scenario_scores.append(score)
        
        avg_score = sum(scenario_scores) / len(scenario_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_scenario_idx = scenario_idx
    
    if best_scenario_idx is not None:
        best_scenario = scenario_names[best_scenario_idx]
        best_analysis = analysis_results[best_scenario_idx]
        
        report.append(f"### Recommended Configuration\n\n")
        report.append(f"Based on the analysis, the recommended configuration is **{best_scenario}** with the following parameters:\n\n")
        
        if 'recommended_arrival_rate' in best_analysis:
            report.append(f"- **Maximum sustainable arrival rate**: {best_analysis['recommended_arrival_rate']:.2f} customers/minute\n")
        if 'optimal_utilization' in best_analysis:
            report.append(f"- **Target utilization**: {best_analysis['optimal_utilization']:.2f}\n")
        
        report.append("\n### Key Insights\n\n")
        report.append("- The system performance starts to degrade significantly when utilization exceeds the critical point\n")
        report.append("- Higher arrival rates lead to increased waiting times and customer abandonment\n")
        report.append("- Balancing resource utilization and customer experience is crucial for optimal operation\n")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(''.join(report))
    
    return output_file
