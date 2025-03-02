import simpy
import random
import numpy as np
from enum import Enum

class CustomerPriority(Enum):
    LOW = 0
    STANDARD = 1
    PREMIUM = 2
    VIP = 3

class AgentType(Enum):
    JUNIOR = 0
    STANDARD = 1
    SENIOR = 2

class ServiceCategory(Enum):
    INQUIRY = 0  # Simple inquiries
    TECHNICAL = 1  # Technical support
    BILLING = 2  # Billing issues
    COMPLAINT = 3  # Complex complaints

class ServicePlatformSimulation:
    def __init__(self, env, agent_configuration, mean_service_time, service_time_std=2, 
                 enable_skills=True, enable_priority=True, time_dependent=True):
        self.env = env
        self.enable_skills = enable_skills
        self.enable_priority = enable_priority
        self.time_dependent = time_dependent
        
        # Initialize agent pools by type if skills enabled
        if enable_skills:
            self.agent_pools = {
                AgentType.JUNIOR: simpy.PriorityResource(env, capacity=agent_configuration.get(AgentType.JUNIOR, 1)),
                AgentType.STANDARD: simpy.PriorityResource(env, capacity=agent_configuration.get(AgentType.STANDARD, 2)),
                AgentType.SENIOR: simpy.PriorityResource(env, capacity=agent_configuration.get(AgentType.SENIOR, 1)),
            }
            self.total_agents = sum(agent_configuration.values())
        else:
            # Simple model with a single pool
            self.total_agents = sum(agent_configuration.values())
            self.agents = simpy.PriorityResource(env, capacity=self.total_agents)
        
        # Service time parameters
        self.mean_service_time = mean_service_time
        self.service_time_std = service_time_std
        
        # Service time multipliers by category and agent type
        self.service_time_multipliers = {
            ServiceCategory.INQUIRY: 0.7,
            ServiceCategory.TECHNICAL: 1.0,
            ServiceCategory.BILLING: 0.9,
            ServiceCategory.COMPLAINT: 1.4
        }
        
        self.agent_efficiency = {
            AgentType.JUNIOR: 1.3,     # 30% slower
            AgentType.STANDARD: 1.0,   # baseline
            AgentType.SENIOR: 0.8      # 20% faster
        }
        
        # Statistics tracking
        self.waiting_times = []
        self.service_times = []
        self.busy_log = [(0, 0)]
        self.completed_customers = 0
        self.abandoned_customers = 0
        self.max_queue_length = 0
        self.queue_length_log = [(0, 0)]
        
        # Advanced metrics
        self.customer_satisfaction = []  # 1-10 scale
        self.customer_wait_by_priority = {
            CustomerPriority.LOW: [],
            CustomerPriority.STANDARD: [],
            CustomerPriority.PREMIUM: [],
            CustomerPriority.VIP: []
        }
        self.service_times_by_category = {cat: [] for cat in ServiceCategory}
        self.abandonment_by_priority = {priority: 0 for priority in CustomerPriority}
        self.abandonment_by_category = {cat: 0 for cat in ServiceCategory}
        self.hourly_arrivals = [0] * 24
        self.hourly_completions = [0] * 24
        
        # Initialize metrics by agent type
        self.service_times_by_agent_type = {agent_type: [] for agent_type in AgentType}
        self.customers_by_agent_type = {agent_type: 0 for agent_type in AgentType}
        self.utilization_by_agent_type = {agent_type: [] for agent_type in AgentType}

    def get_service_time(self, service_category, agent_type):
        """Calculate service time based on category and agent type."""
        base_time = random.normalvariate(self.mean_service_time, self.service_time_std)
        base_time = max(0.1, base_time)  # Ensure positive
        
        # Apply category multiplier
        category_multiplier = self.service_time_multipliers.get(service_category, 1.0)
        
        # Apply agent efficiency
        agent_multiplier = self.agent_efficiency.get(agent_type, 1.0)
        
        return base_time * category_multiplier * agent_multiplier

    def calculate_satisfaction(self, wait_time, service_time, service_category, priority):
        """Calculate customer satisfaction (1-10) based on experience."""
        # Base satisfaction starts at 8
        satisfaction = 8
        
        # Deduct for waiting time (more impact for higher priority customers)
        priority_factor = priority.value + 1
        wait_penalty = min(4, wait_time / (5 * (5 - priority_factor)))
        satisfaction -= wait_penalty
        
        # Deduct for long service times
        expected_service = self.mean_service_time * self.service_time_multipliers.get(service_category, 1.0)
        if service_time > expected_service * 1.2:  # 20% longer than expected
            satisfaction -= 1
        
        # Add bonus for quick service
        if wait_time < 1 and service_time < expected_service:
            satisfaction += 1
            
        # Ensure within 1-10 range and round to nearest 0.5
        satisfaction = max(1, min(10, satisfaction))
        satisfaction = round(satisfaction * 2) / 2
        
        return satisfaction

    def get_appropriate_agent_type(self, service_category):
        """Determine the most appropriate agent type for the service category."""
        if service_category == ServiceCategory.INQUIRY:
            return random.choices(
                [AgentType.JUNIOR, AgentType.STANDARD], 
                weights=[0.7, 0.3], 
                k=1
            )[0]
        elif service_category == ServiceCategory.TECHNICAL:
            return random.choices(
                [AgentType.STANDARD, AgentType.SENIOR], 
                weights=[0.6, 0.4], 
                k=1
            )[0]
        elif service_category == ServiceCategory.BILLING:
            return random.choices(
                [AgentType.JUNIOR, AgentType.STANDARD, AgentType.SENIOR], 
                weights=[0.3, 0.6, 0.1], 
                k=1
            )[0]
        elif service_category == ServiceCategory.COMPLAINT:
            return random.choices(
                [AgentType.STANDARD, AgentType.SENIOR], 
                weights=[0.3, 0.7], 
                k=1
            )[0]
        return AgentType.STANDARD  # Default
    
    def get_current_hour(self):
        """Get the current hour based on simulation time (assuming 24-hour day)."""
        return int(self.env.now / 60) % 24

    def customer_process(self, arrival_time, priority=CustomerPriority.STANDARD, 
                         service_category=ServiceCategory.INQUIRY, patience=None):
        """Simulate a customer arriving and being served."""
        # Record arrival hour
        arrival_hour = self.get_current_hour()
        self.hourly_arrivals[arrival_hour] += 1
        
        # Track queue length
        current_queue = sum(len(pool.queue) for pool in self.agent_pools.values()) if self.enable_skills else len(self.agents.queue)
        self.max_queue_length = max(self.max_queue_length, current_queue)
        self.queue_length_log.append((self.env.now, current_queue))
        
        # Determine priority value for queue (lower value = higher priority)
        priority_value = -priority.value if self.enable_priority else 0
        
        # Determine which agent type to request
        if self.enable_skills:
            agent_type = self.get_appropriate_agent_type(service_category)
            agent_pool = self.agent_pools[agent_type]
            request = agent_pool.request(priority=priority_value)
        else:
            agent_type = AgentType.STANDARD  # Default for no skills model
            request = self.agents.request(priority=priority_value)
        
        # Request for service with timeout if customer has patience limit
        if patience:
            patience_timeout = self.env.timeout(patience)
            result = yield patience_timeout | request
            
            if patience_timeout in result:  # Customer abandoned
                self.abandoned_customers += 1
                self.abandonment_by_priority[priority] += 1
                self.abandonment_by_category[service_category] += 1
                
                # Remove request from queue if it's there
                if self.enable_skills:
                    if request in agent_pool.queue:
                        agent_pool.queue.remove(request)
                else:
                    if request in self.agents.queue:
                        self.agents.queue.remove(request)
                return
        else:
            # Customer has infinite patience
            yield request
        
        # Customer is now being served
        start_time = self.env.now
        wait_time = start_time - arrival_time
        
        # Update busy log and waiting times
        if self.enable_skills:
            busy_count = sum(len(pool.users) for pool in self.agent_pools.values())
            self.busy_log.append((start_time, busy_count))
        else:
            self.busy_log.append((start_time, len(self.agents.users)))
            
        self.waiting_times.append(wait_time)
        self.customer_wait_by_priority[priority].append(wait_time)

        # Calculate service time based on category and agent type
        service_time = self.get_service_time(service_category, agent_type)
        self.service_times.append(service_time)
        self.service_times_by_category[service_category].append(service_time)
        self.service_times_by_agent_type[agent_type].append(service_time)
        
        # Perform service
        yield self.env.timeout(service_time)

        # Service completed
        self.completed_customers += 1
        self.customers_by_agent_type[agent_type] += 1
        
        # Record completion hour
        completion_hour = self.get_current_hour()
        self.hourly_completions[completion_hour] += 1
        
        # Calculate and store customer satisfaction
        satisfaction = self.calculate_satisfaction(wait_time, service_time, service_category, priority)
        self.customer_satisfaction.append(satisfaction)
        
        # Update busy log
        if self.enable_skills:
            busy_count = sum(len(pool.users) for pool in self.agent_pools.values()) - 1
            self.busy_log.append((self.env.now, busy_count))
            
            # Update utilization for this agent type
            users = len(self.agent_pools[agent_type].users) - 1
            capacity = self.agent_pools[agent_type].capacity
            self.utilization_by_agent_type[agent_type].append(
                (self.env.now, users / capacity if capacity > 0 else 0)
            )
        else:
            self.busy_log.append((self.env.now, len(self.agents.users) - 1))
        
        # Track queue length after service
        current_queue = sum(len(pool.queue) for pool in self.agent_pools.values()) if self.enable_skills else len(self.agents.queue)
        self.queue_length_log.append((self.env.now, current_queue))
        
        if self.enable_skills:
            agent_pool.release(request)
        else:
            self.agents.release(request)

    def get_time_dependent_rate(self, base_rate):
        """Adjust arrival rate based on time of day."""
        if not self.time_dependent:
            return base_rate
            
        hour = self.get_current_hour()
        
        # Define hourly patterns (busier during business hours)
        hourly_factors = [
            0.2,  # 12 AM - very low traffic
            0.1, 0.1, 0.1, 0.1, 0.2,  # 1 AM - 5 AM
            0.3, 0.6, 0.9,  # 6 AM - 8 AM (increasing)
            1.2, 1.5, 1.8,  # 9 AM - 11 AM (busy)
            2.0,  # 12 PM (peak)
            1.8, 1.5, 1.3,  # 1 PM - 3 PM
            1.4, 1.5, 1.3,  # 4 PM - 6 PM
            1.0, 0.8, 0.6,  # 7 PM - 9 PM
            0.4, 0.3, 0.2  # 10 PM - 12 AM
        ]
        
        return base_rate * hourly_factors[hour]

    def customer_arrival_process(self, base_arrival_rate, max_patience=None):
        """Generate customer arrivals."""
        while True:
            # Adjust arrival rate based on time of day if enabled
            arrival_rate = self.get_time_dependent_rate(base_arrival_rate)
            
            # Generate next arrival
            inter_arrival_time = random.expovariate(arrival_rate)
            yield self.env.timeout(inter_arrival_time)
            arrival_time = self.env.now
            
            # Assign customer attributes
            # Priority based on distribution
            priority = random.choices(
                [CustomerPriority.LOW, CustomerPriority.STANDARD, 
                 CustomerPriority.PREMIUM, CustomerPriority.VIP],
                weights=[0.2, 0.6, 0.15, 0.05],
                k=1
            )[0]
            
            # Service category based on distribution
            category = random.choices(
                list(ServiceCategory),
                weights=[0.4, 0.3, 0.2, 0.1],
                k=1
            )[0]
            
            # Patience depends on priority and has randomness
            patience = None
            if max_patience:
                # Higher priority customers have more patience on average
                priority_factor = 0.8 + (priority.value * 0.1)  # 0.8, 0.9, 1.0, 1.1
                patience = random.expovariate(1/(max_patience * priority_factor))
                
            self.env.process(self.customer_process(
                arrival_time, 
                priority=priority,
                service_category=category,
                patience=patience
            ))