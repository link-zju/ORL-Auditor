from re import M
from unittest import result
import numpy as np
from .my_auditor import MyAuditor
from scipy import stats
class ConstructDetectionFeature():
    def __init__(self, metric_list, preprecess_flag=False):
        self.preprecess_flag = preprecess_flag

    def preprecessing(self, x1, x2):
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        return x1, x2

    def L1_distance(self):
        return np.abs(self.x1 - self.x2).sum()

    def L2_distance(self):
        return np.linalg.norm(self.x1 - self.x2)
    
    def output_results(self, x1, x2, action_dim):
        sample_num = 2500

        
        self.x1, self.x2 = x1[:sample_num, -action_dim:], x2[:sample_num, -action_dim:]
        results = {}
        results['l1_dis'] = self.L1_distance()
        results['l2_dis'] = self.L2_distance()



        initial_dim_division = {'c0': np.zeros(2),
                                'c1': np.zeros(2)
                                } 
        initial_dim_division_space = 1

        return results