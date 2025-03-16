import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def load_call_center_data(filepath):
    """Load call center data from CSV file"""
    data = pd.read_csv(filepath)
    return data

def process_call_data(data):
    """Process and extract key metrics from call center data"""
    # Convert time columns to datetime
    for col in ['date', 'call_started', 'call_answered', 'call_ended']:
        if col in data.columns:
            if col == 'date':
                data[col] = pd.to_datetime(data[col])
            else:
                # Handle time columns - combine with date if needed
                try:
                    data[col] = pd.to_datetime(data[col])
                except:
                    # If time format only
                    data[col] = pd.to_datetime(data['date'].dt.date.astype(str) + ' ' + data[col])

    # Extract metrics
    metrics = {}
    
    # Service time stats
    metrics['mean_service_time'] = data['service_length'].mean()
    metrics['std_service_time'] = data['service_length'].std()
    
    # Wait time stats
    metrics['mean_wait_time'] = data['wait_length'].mean()
    metrics['median_wait_time'] = data['wait_length'].median()
    metrics['max_wait_time'] = data['wait_length'].max()
    
    # Service standard compliance
    if 'meets_standard' in data.columns:
        metrics['service_standard_rate'] = data['meets_standard'].mean()
    
    # Calculate call arrival patterns by hour
    if 'call_started' in data.columns and pd.api.types.is_datetime64_dtype(data['call_started']):
        data['hour'] = data['call_started'].dt.hour
        hourly_arrivals = data.groupby('hour').size()
        metrics['hourly_pattern'] = hourly_arrivals / hourly_arrivals.sum()
    
    # Calculate average calls per day
    if 'date' in data.columns:
        calls_per_day = data.groupby(data['date'].dt.date).size()
        metrics['avg_daily_calls'] = calls_per_day.mean()
        metrics['avg_arrival_rate'] = calls_per_day.mean() / (8 * 60)  # calls per minute assuming 8-hour day
    elif 'daily_caller' in data.columns:
        # If we have daily call count in the data
        metrics['avg_daily_calls'] = data['daily_caller'].mean()
        metrics['avg_arrival_rate'] = data['daily_caller'].mean() / (8 * 60)
    
    return metrics

def visualize_call_data(data):
    """Create visualizations of key call center metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot wait time distribution
    axes[0, 0].hist(data['wait_length'], bins=30)
    axes[0, 0].set_title('Wait Time Distribution')
    axes[0, 0].set_xlabel('Wait Time (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Plot service time distribution
    axes[0, 1].hist(data['service_length'], bins=30)
    axes[0, 1].set_title('Service Time Distribution')
    axes[0, 1].set_xlabel('Service Time (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot hourly call pattern if time data is available
    if 'call_started' in data.columns and pd.api.types.is_datetime64_dtype(data['call_started']):
        hourly_pattern = data.groupby(data['call_started'].dt.hour).size()
        axes[1, 0].bar(hourly_pattern.index, hourly_pattern.values)
        axes[1, 0].set_title('Hourly Call Pattern')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Number of Calls')
    
    # Plot daily call volume if date data is available
    if 'date' in data.columns:
        daily_volume = data.groupby(data['date'].dt.date).size()
        axes[1, 1].plot(daily_volume.index, daily_volume.values)
        axes[1, 1].set_title('Daily Call Volume')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Number of Calls')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('call_data_analysis.png')
    plt.close()
    
    return 'call_data_analysis.png'

def validate_simulation_against_data(sim_results, real_metrics):
    """Compare simulation results with real data metrics"""
    comparison = {}
    
    # Compare key metrics
    if 'avg_waiting_time' in sim_results and 'mean_wait_time' in real_metrics:
        comparison['wait_time_diff'] = sim_results['avg_waiting_time'] - real_metrics['mean_wait_time']
        comparison['wait_time_pct_error'] = (comparison['wait_time_diff'] / real_metrics['mean_wait_time']) * 100
    
    if 'avg_service_time' in sim_results and 'mean_service_time' in real_metrics:
        comparison['service_time_diff'] = sim_results['avg_service_time'] - real_metrics['mean_service_time']
        comparison['service_time_pct_error'] = (comparison['service_time_diff'] / real_metrics['mean_service_time']) * 100
    
    # Compare throughput if available
    if 'throughput' in sim_results and 'avg_arrival_rate' in real_metrics:
        comparison['throughput_diff'] = sim_results['throughput'] - real_metrics['avg_arrival_rate']
        comparison['throughput_pct_error'] = (comparison['throughput_diff'] / real_metrics['avg_arrival_rate']) * 100
    
    return comparison
