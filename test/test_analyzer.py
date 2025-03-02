import unittest
import simpy
from src.simulation import ServicePlatformSimulation
from src.analyzer import calculate_metrics

class TestAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.env = simpy.Environment()
        self.sim = ServicePlatformSimulation(self.env, num_agents=2, mean_service_time=5)
    
    def test_calculate_metrics_empty(self):
        """Test metrics calculation with no customers."""
        metrics = calculate_metrics(self.sim, 10)
        
        self.assertEqual(metrics['avg_waiting_time'], 0)
        self.assertEqual(metrics['utilization'], 0)
        self.assertEqual(metrics['completed_customers'], 0)
    
    def test_calculate_metrics_with_customers(self):
        """Test metrics calculation with customers."""
        # Manually add some data
        self.sim.waiting_times = [2, 4, 6]
        self.sim.service_times = [5, 7, 4]
        self.sim.busy_log = [(0, 0), (2, 1), (7, 2), (12, 1), (16, 0)]
        self.sim.completed_customers = 3
        
        metrics = calculate_metrics(self.sim, 20)
        
        self.assertEqual(metrics['avg_waiting_time'], 4)
        self.assertGreater(metrics['utilization'], 0)
        self.assertEqual(metrics['completed_customers'], 3)
    
    def test_calculate_queue_metrics(self):
        """Test queue metrics calculation."""
        # Set up simulation with queue data
        self.sim.max_queue_length = 3
        
        metrics = calculate_metrics(self.sim, 10)
        
        self.assertEqual(metrics['max_queue_length'], 3)

if __name__ == '__main__':
    unittest.main()
