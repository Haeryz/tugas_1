import numpy as np
import pandas as pd
from scipy import stats

def calculate_metrics(sim, sim_time):
    """Calculate performance metrics from simulation results."""
    metrics = {}
    
    # Basic metrics
    if len(sim.waiting_times) > 0:
        metrics['avg_waiting_time'] = np.mean(sim.waiting_times)
        metrics['median_waiting_time'] = np.median(sim.waiting_times)
        metrics['p95_waiting_time'] = np.percentile(sim.waiting_times, 95)
        metrics['max_waiting_time'] = np.max(sim.waiting_times)
    else:
        metrics['avg_waiting_time'] = 0
        metrics['median_waiting_time'] = 0
        metrics['p95_waiting_time'] = 0
        metrics['max_waiting_time'] = 0
    
    if len(sim.service_times) > 0:
        metrics['avg_service_time'] = np.mean(sim.service_times)
        metrics['median_service_time'] = np.median(sim.service_times)
    else:
        metrics['avg_service_time'] = 0
        metrics['median_service_time'] = 0

    # Calculate utilization
    if len(sim.busy_log) > 1:
        # Calculate area under the curve
        utilization_time = 0
        for i in range(len(sim.busy_log) - 1):
            t1, busy1 = sim.busy_log[i]
            t2, busy2 = sim.busy_log[i + 1]
            if t1 != t2:  # Avoid division by zero
                utilization_time += (t2 - t1) * busy1
        
        metrics['utilization'] = utilization_time / (sim_time * sim.total_agents) if sim.total_agents > 0 else 0
    else:
        metrics['utilization'] = 0
    
    # Advanced agent utilization metrics if skills are enabled
    if hasattr(sim, 'enable_skills') and sim.enable_skills:
        for agent_type in sim.utilization_by_agent_type:
            ut_log = sim.utilization_by_agent_type[agent_type]
            if len(ut_log) > 1:
                agent_utilization_time = 0
                for i in range(len(ut_log) - 1):
                    t1, u1 = ut_log[i]
                    t2, _ = ut_log[i + 1]
                    if t1 != t2:
                        agent_utilization_time += (t2 - t1) * u1
                
                metrics[f'utilization_{agent_type.name}'] = agent_utilization_time / sim_time
            else:
                metrics[f'utilization_{agent_type.name}'] = 0
    
    # Customer metrics
    metrics['completed_customers'] = sim.completed_customers
    metrics['abandoned_customers'] = sim.abandoned_customers
    total_customers = sim.completed_customers + sim.abandoned_customers
    metrics['abandonment_rate'] = sim.abandoned_customers / total_customers if total_customers > 0 else 0
    
    # Queue metrics
    metrics['max_queue_length'] = sim.max_queue_length
    
    # Customer satisfaction
    if len(sim.customer_satisfaction) > 0:
        metrics['avg_customer_satisfaction'] = np.mean(sim.customer_satisfaction)
        metrics['pct_satisfied'] = np.mean([1 if s >= 7 else 0 for s in sim.customer_satisfaction])
    else:
        metrics['avg_customer_satisfaction'] = 0
        metrics['pct_satisfied'] = 0
    
    # Priority-based metrics
    for priority in sim.customer_wait_by_priority:
        wait_times = sim.customer_wait_by_priority[priority]
        if wait_times:
            metrics[f'avg_wait_{priority.name}'] = np.mean(wait_times)
        else:
            metrics[f'avg_wait_{priority.name}'] = 0
    
    # Calculate priority-specific abandonment rates
    for priority in sim.abandonment_by_priority:
        abandons = sim.abandonment_by_priority[priority]
        total_by_priority = sum(1 for t in sim.customer_wait_by_priority[priority]) + abandons
        metrics[f'abandonment_rate_{priority.name}'] = abandons / total_by_priority if total_by_priority > 0 else 0
    
    # Service category metrics
    for category in sim.service_times_by_category:
        times = sim.service_times_by_category[category]
        if times:
            metrics[f'avg_service_time_{category.name}'] = np.mean(times)
        else:
            metrics[f'avg_service_time_{category.name}'] = 0
    
    # Agent type metrics
    if hasattr(sim, 'customers_by_agent_type'):
        for agent_type in sim.customers_by_agent_type:
            metrics[f'customers_handled_{agent_type.name}'] = sim.customers_by_agent_type[agent_type]
            
            times = sim.service_times_by_agent_type[agent_type]
            if times:
                metrics[f'avg_service_time_{agent_type.name}'] = np.mean(times)
            else:
                metrics[f'avg_service_time_{agent_type.name}'] = 0
    
    # Time-of-day metrics
    metrics['hourly_arrivals'] = sim.hourly_arrivals
    metrics['hourly_completions'] = sim.hourly_completions
    metrics['peak_hour'] = np.argmax(sim.hourly_arrivals)
    
    # Calculate statistical confidence intervals for key metrics
    if len(sim.waiting_times) >= 30:  # Enough samples for meaningful CI
        ci = stats.t.interval(
            0.95,  # 95% confidence
            len(sim.waiting_times)-1,
            loc=np.mean(sim.waiting_times),
            scale=stats.sem(sim.waiting_times)
        )
        metrics['wait_time_ci_low'] = ci[0]
        metrics['wait_time_ci_high'] = ci[1]
    
    # Calculate throughput
    metrics['throughput'] = sim.completed_customers / sim_time  # customers per minute
    
    return metrics

def create_metrics_dataframe(results_list, scenarios, arrival_rates):
    """Create a pandas DataFrame for easier analysis of multiple scenarios."""
    data = []
    
    for scenario_idx, scenario_results in enumerate(results_list):
        scenario_name = scenarios[scenario_idx]
        for rate_idx, result in enumerate(scenario_results):
            rate = arrival_rates[rate_idx]
            row = {'scenario': scenario_name, 'arrival_rate': rate}
            row.update(result)  # Add all metrics
            data.append(row)
    
    return pd.DataFrame(data)

def calculate_optimal_staffing(df, target_wait=5, target_util=0.8):
    """Analyze results to recommend optimal staffing levels."""
    # Group by number of agents
    if 'num_agents' in df.columns:
        grouped = df.groupby('num_agents')
        
        results = []
        for num_agents, group in grouped:
            avg_wait = group['avg_waiting_time'].mean()
            avg_util = group['utilization'].mean()
            avg_abandon = group['abandonment_rate'].mean()
            
            # Score based on how close to targets
            wait_score = max(0, 1 - abs(avg_wait - target_wait)/target_wait)
            util_score = max(0, 1 - abs(avg_util - target_util)/target_util)
            abandon_penalty = avg_abandon * 2  # Penalize abandonment
            
            total_score = wait_score + util_score - abandon_penalty
            
            results.append({
                'num_agents': num_agents,
                'avg_wait': avg_wait,
                'avg_util': avg_util,
                'avg_abandon': avg_abandon,
                'score': total_score
            })
        
        # Sort by score and return recommendations
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:3]  # Return top 3 recommendations
    
    return None