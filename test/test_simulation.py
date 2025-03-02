import unittest
import simpy
from src.simulation import ServicePlatformSimulation

class TestServicePlatformSimulation(unittest.TestCase):
    
    def setUp(self):
        self.env = simpy.Environment()
        self.sim = ServicePlatformSimulation(self.env, num_agents=2, mean_service_time=5)
    
    def test_init(self):
        """Test initialization of simulation object."""
        self.assertEqual(self.sim.agents.capacity, 2)
        self.assertEqual(self.sim.mean_service_time, 5)
        self.assertEqual(len(self.sim.waiting_times), 0)
        self.assertEqual(len(self.sim.busy_log), 1)
    
    def test_customer_process(self):
        """Test customer process with short simulation."""
        # Add one customer
        self.env.process(self.sim.customer_process(arrival_time=0))
        self.env.run(until=10)
        
        # Check that customer was processed
        self.assertEqual(len(self.sim.waiting_times), 1)
        self.assertEqual(self.sim.completed_customers, 1)
    
    def test_customer_arrival_process(self):
        """Test customer arrivals."""
        # Start arrival process with high rate to ensure customers arrive
        self.env.process(self.sim.customer_arrival_process(arrival_rate=1.0))
        self.env.run(until=5)
        
        # Check that customers arrived and some were processed
        self.assertGreater(len(self.sim.waiting_times), 0)
    
    def test_customer_abandonment(self):
        """Test customer abandonment with limited patience."""
        # Create sim with 1 agent to force queuing
        env = simpy.Environment()
        sim = ServicePlatformSimulation(env, num_agents=1, mean_service_time=10)
        
        # Add customers with very short patience
        for i in range(3):
            env.process(sim.customer_process(arrival_time=i, patience=0.1))
        
        env.run(until=20)
        
        # Check that some customers abandoned
        self.assertGreater(sim.abandoned_customers, 0)

if __name__ == '__main__':
    unittest.main()
