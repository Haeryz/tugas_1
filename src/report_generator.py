import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os
from simulation import AgentType, CustomerPriority, ServiceCategory

def ensure_output_dir(base_dir='../output'):
    """Ensure output directories exist"""
    dirs = [
        base_dir,
        f"{base_dir}/tables",
        f"{base_dir}/figures",
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return base_dir

def generate_base_scenario_table(metrics_df, output_dir=None):
    """Generate a table for base scenario results"""
    base_df = metrics_df[metrics_df['scenario'] == 'Infinite Patience'].copy()
    
    # Create a formatted table
    table_df = base_df[['arrival_rate', 'avg_waiting_time', 'median_waiting_time', 
                       'utilization', 'completed_customers', 'throughput']].copy()
    
    # Format the columns for better display
    table_df['arrival_rate'] = table_df['arrival_rate'].map(lambda x: f"{x:.1f}")
    table_df['avg_waiting_time'] = table_df['avg_waiting_time'].map(lambda x: f"{x:.2f}")
    table_df['median_waiting_time'] = table_df['median_waiting_time'].map(lambda x: f"{x:.2f}")
    table_df['utilization'] = table_df['utilization'].map(lambda x: f"{x:.2%}")
    table_df['throughput'] = table_df['throughput'].map(lambda x: f"{x:.2f}")
    
    # Rename columns for better presentation
    table_df.columns = ['Arrival Rate (customers/min)', 'Avg. Wait Time (min)', 
                      'Median Wait Time (min)', 'Utilization Rate', 
                      'Completed Customers', 'Throughput (customers/min)']
    
    # Save to CSV
    if output_dir:
        table_path = f"{output_dir}/tables/base_scenario_results.csv"
        table_df.to_csv(table_path, index=False)
        print(f"Base scenario table saved to {table_path}")
    
    return table_df

def plot_arrival_rate_effects(metrics_df, output_dir=None):
    """Create visualizations for the effect of arrival rates"""
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Filter for infinite patience scenarios to focus on arrival rate effects
    base_df = metrics_df[metrics_df['scenario'] == 'Infinite Patience'].copy()
    
    # 1. Waiting Time vs Arrival Rate
    sns.lineplot(x='arrival_rate', y='avg_waiting_time', 
                 marker='o', linewidth=2, ax=axes[0, 0], data=base_df)
    axes[0, 0].set_title('Average Waiting Time vs Arrival Rate')
    axes[0, 0].set_xlabel('Arrival Rate (customers/min)')
    axes[0, 0].set_ylabel('Average Waiting Time (min)')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Utilization vs Arrival Rate
    sns.lineplot(x='arrival_rate', y='utilization', 
                 marker='o', linewidth=2, ax=axes[0, 1], data=base_df)
    axes[0, 1].set_title('System Utilization vs Arrival Rate')
    axes[0, 1].set_xlabel('Arrival Rate (customers/min)')
    axes[0, 1].set_ylabel('Utilization Rate')
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Completed Customers vs Arrival Rate
    sns.lineplot(x='arrival_rate', y='completed_customers', 
                 marker='o', linewidth=2, ax=axes[1, 0], data=base_df)
    axes[1, 0].set_title('Completed Customers vs Arrival Rate')
    axes[1, 0].set_xlabel('Arrival Rate (customers/min)')
    axes[1, 0].set_ylabel('Number of Completed Customers')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Throughput vs Arrival Rate
    sns.lineplot(x='arrival_rate', y='throughput', 
                 marker='o', linewidth=2, ax=axes[1, 1], data=base_df)
    axes[1, 1].set_title('System Throughput vs Arrival Rate')
    axes[1, 1].set_xlabel('Arrival Rate (customers/min)')
    axes[1, 1].set_ylabel('Throughput (customers/min)')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        fig_path = f"{output_dir}/figures/arrival_rate_effects.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Arrival rate effects figure saved to {fig_path}")
        plt.close()
    else:
        plt.show()
    
    return fig

def plot_patience_effects(metrics_df, output_dir=None):
    """Create visualizations for the effect of customer patience"""
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Prepare data by pivoting for comparison
    scenarios = metrics_df['scenario'].unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Abandonment Rate vs Arrival Rate
    for i, scenario in enumerate(scenarios):
        scenario_df = metrics_df[metrics_df['scenario'] == scenario]
        if 'abandonment_rate' in scenario_df.columns:
            sns.lineplot(x='arrival_rate', y='abandonment_rate', 
                         marker='o', linewidth=2, ax=axes[0, 0], 
                         label=scenario, color=colors[i % len(colors)],
                         data=scenario_df)
    
    axes[0, 0].set_title('Abandonment Rate vs Arrival Rate')
    axes[0, 0].set_xlabel('Arrival Rate (customers/min)')
    axes[0, 0].set_ylabel('Abandonment Rate')
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Completed Customers by Scenario
    for i, scenario in enumerate(scenarios):
        scenario_df = metrics_df[metrics_df['scenario'] == scenario]
        sns.lineplot(x='arrival_rate', y='completed_customers', 
                     marker='o', linewidth=2, ax=axes[0, 1], 
                     label=scenario, color=colors[i % len(colors)],
                     data=scenario_df)
    
    axes[0, 1].set_title('Completed Customers by Scenario')
    axes[0, 1].set_xlabel('Arrival Rate (customers/min)')
    axes[0, 1].set_ylabel('Number of Completed Customers')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Average Waiting Time Comparison
    for i, scenario in enumerate(scenarios):
        scenario_df = metrics_df[metrics_df['scenario'] == scenario]
        sns.lineplot(x='arrival_rate', y='avg_waiting_time', 
                     marker='o', linewidth=2, ax=axes[1, 0], 
                     label=scenario, color=colors[i % len(colors)],
                     data=scenario_df)
    
    axes[1, 0].set_title('Average Waiting Time by Scenario')
    axes[1, 0].set_xlabel('Arrival Rate (customers/min)')
    axes[1, 0].set_ylabel('Average Waiting Time (min)')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()
    
    # 4. System Utilization Comparison
    for i, scenario in enumerate(scenarios):
        scenario_df = metrics_df[metrics_df['scenario'] == scenario]
        sns.lineplot(x='arrival_rate', y='utilization', 
                     marker='o', linewidth=2, ax=axes[1, 1], 
                     label=scenario, color=colors[i % len(colors)],
                     data=scenario_df)
    
    axes[1, 1].set_title('System Utilization by Scenario')
    axes[1, 1].set_xlabel('Arrival Rate (customers/min)')
    axes[1, 1].set_ylabel('Utilization Rate')
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        fig_path = f"{output_dir}/figures/patience_effects.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Patience effects figure saved to {fig_path}")
        plt.close()
    else:
        plt.show()
    
    return fig

def create_agent_utilization_table(metrics_df, output_dir=None):
    """Create a table showing utilization by agent type"""
    # Create empty table structure
    agent_types = [AgentType.JUNIOR, AgentType.STANDARD, AgentType.SENIOR]
    agent_columns = [f"utilization_{agent_type.name}" for agent_type in agent_types]
    
    # Filter columns that exist
    available_columns = [col for col in agent_columns if col in metrics_df.columns]
    
    if not available_columns:
        print("Agent utilization data not available in metrics")
        return None
    
    # Create and format table
    table_df = metrics_df[['scenario', 'arrival_rate'] + available_columns].copy()
    
    for col in available_columns:
        table_df[col] = table_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    
    # Rename columns for better presentation
    display_columns = {col: f"{col.split('_')[1]} Utilization" for col in available_columns}
    display_columns.update({'scenario': 'Scenario', 'arrival_rate': 'Arrival Rate'})
    table_df.rename(columns=display_columns, inplace=True)
    
    # Save to CSV
    if output_dir:
        table_path = f"{output_dir}/tables/agent_utilization.csv"
        table_df.to_csv(table_path, index=False)
        print(f"Agent utilization table saved to {table_path}")
    
    return table_df

def plot_system_performance(metrics_df, output_dir=None):
    """Create advanced visualizations for system performance analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Waiting Time Distribution (if we have raw waiting times, use kernel density estimate)
    if 'avg_waiting_time' in metrics_df.columns and 'scenario' in metrics_df.columns:
        scenarios = metrics_df['scenario'].unique()
        for scenario in scenarios:
            scenario_df = metrics_df[metrics_df['scenario'] == scenario]
            sns.scatterplot(x='arrival_rate', y='avg_waiting_time', 
                           data=scenario_df, ax=axes[0, 0], label=scenario, s=100)
        
        # Add best fit lines
        for scenario in scenarios:
            scenario_df = metrics_df[metrics_df['scenario'] == scenario]
            x = scenario_df['arrival_rate'].values
            y = scenario_df['avg_waiting_time'].values
            if len(x) > 1:  # Need at least 2 points for regression
                # Add a polynomial fit (quadratic)
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                axes[0, 0].plot(x_line, p(x_line), '--', 
                               label=f"{scenario} trend")
        
        axes[0, 0].set_title('Waiting Time vs Arrival Rate (with trends)')
        axes[0, 0].set_xlabel('Arrival Rate (customers/min)')
        axes[0, 0].set_ylabel('Average Waiting Time (min)')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()
    
    # 2. Waiting Time vs Utilization (scatter with trendline)
    if 'utilization' in metrics_df.columns and 'avg_waiting_time' in metrics_df.columns:
        scenarios = metrics_df['scenario'].unique()
        for scenario in scenarios:
            scenario_df = metrics_df[metrics_df['scenario'] == scenario]
            sns.scatterplot(x='utilization', y='avg_waiting_time', 
                           data=scenario_df, ax=axes[0, 1], label=scenario, s=100)
        
        # Add best fit lines and annotations
        for scenario in scenarios:
            scenario_df = metrics_df[metrics_df['scenario'] == scenario]
            x = scenario_df['utilization'].values
            y = scenario_df['avg_waiting_time'].values
            if len(x) > 1:  # Need at least 2 points for regression
                # Add exponential fit for waiting time vs utilization
                # This relationship often follows an exponential pattern
                from scipy.optimize import curve_fit
                
                def exp_func(x, a, b, c):
                    return a * np.exp(b * x) + c
                
                try:
                    popt, _ = curve_fit(exp_func, x, y, p0=[0.1, 5, 0])
                    x_line = np.linspace(min(x), max(x), 100)
                    axes[0, 1].plot(x_line, exp_func(x_line, *popt), '--',
                                  label=f"{scenario} trend")
                except:
                    # If exponential fit fails, use polynomial
                    z = np.polyfit(x, y, 2)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(x), max(x), 100)
                    axes[0, 1].plot(x_line, p(x_line), '--', 
                                  label=f"{scenario} trend")
        
        axes[0, 1].set_title('Waiting Time vs System Utilization')
        axes[0, 1].set_xlabel('Utilization Rate')
        axes[0, 1].set_ylabel('Average Waiting Time (min)')
        axes[0, 1].xaxis.set_major_formatter(PercentFormatter(1.0))
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend()
    
    # 3. Customer Throughput vs Arrival Rate
    if 'throughput' in metrics_df.columns:
        scenarios = metrics_df['scenario'].unique()
        for scenario in scenarios:
            scenario_df = metrics_df[metrics_df['scenario'] == scenario]
            sns.lineplot(x='arrival_rate', y='throughput', 
                        data=scenario_df, ax=axes[1, 0], 
                        marker='o', label=scenario, linewidth=2)
        
        # Add reference line (throughput = arrival rate when no abandonment)
        x_vals = metrics_df['arrival_rate'].unique()
        x_vals.sort()
        axes[1, 0].plot(x_vals, x_vals, 'k--', alpha=0.3, label='Max Theoretical')
        
        axes[1, 0].set_title('System Throughput vs Arrival Rate')
        axes[1, 0].set_xlabel('Arrival Rate (customers/min)')
        axes[1, 0].set_ylabel('Throughput (customers/min)')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()
    
    # 4. Abandonment Rate and Utilization Combined View
    if 'abandonment_rate' in metrics_df.columns and 'utilization' in metrics_df.columns:
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        scenarios = metrics_df['scenario'].unique()
        for i, scenario in enumerate(scenarios):
            scenario_df = metrics_df[metrics_df['scenario'] == scenario]
            if 'abandonment_rate' in scenario_df.columns:
                sns.lineplot(x='arrival_rate', y='abandonment_rate', 
                            data=scenario_df, ax=ax1,
                            marker='o', label=f"{scenario} - Abandonment", 
                            color=f'C{i}', linewidth=2)
                sns.lineplot(x='arrival_rate', y='utilization', 
                            data=scenario_df, ax=ax2,
                            marker='s', label=f"{scenario} - Utilization", 
                            color=f'C{i}', linewidth=2, linestyle='--')
        
        ax1.set_title('Abandonment Rate and Utilization vs Arrival Rate')
        ax1.set_xlabel('Arrival Rate (customers/min)')
        ax1.set_ylabel('Abandonment Rate')
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.set_ylabel('Utilization Rate')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.get_legend().remove()  # Remove duplicate legend
        
        ax1.grid(alpha=0.3)
        
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        fig_path = f"{output_dir}/figures/system_performance.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"System performance figure saved to {fig_path}")
        plt.close()
    else:
        plt.show()
    
    return fig

def generate_all_reports(metrics_df):
    """Generate all report elements"""
    output_dir = ensure_output_dir()
    
    print("Generating report elements...")
    
    # Generate tables
    generate_base_scenario_table(metrics_df, output_dir)
    create_agent_utilization_table(metrics_df, output_dir)
    
    # Generate figures
    plot_arrival_rate_effects(metrics_df, output_dir)
    plot_patience_effects(metrics_df, output_dir)
    plot_system_performance(metrics_df, output_dir)
    
    print(f"All report elements generated in {output_dir}")

if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path to import modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # If metrics data exists, load and use it
    metrics_file = '../output/simulation_metrics.csv'
    if os.path.exists(metrics_file):
        print(f"Loading metrics from {metrics_file}")
        metrics_df = pd.read_csv(metrics_file)
        generate_all_reports(metrics_df)
    else:
        print("No metrics file found. Please run the simulation first.")
