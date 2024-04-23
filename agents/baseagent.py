from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    BaseAgent abstract class defines the required interfaces and for the agent to
    interact with enviornment in the correct way. 
    """
    def __init__(self, name, reward_function, environment, sub_agents = None):
        """
        Initializes the agent object.

        Args:
            name (str): 
                The name of the agent, also used in the environment class.
            reward_function (function): 
                The reward function attached to the agent, which is called by the environment class.
            environment (object): 
                The environment with which the agent interacts.
            sub_agents (list, optional): 
                Sub-agents for multi-agent reinforcement learning (MARL). Defaults to None.
        """
        self.name = name # Name of agent that is also used in enviornment class
        self.reward_function = reward_function # Attached reward function called by enviornment class
        self.env = environment # Environment which agent is interacting with 
        self.sub_agents = None #TBD for MARL
 
    @abstractmethod
    def _act(self,step_type):
        """
        Function that calls the enviornments.step(action)
        
        Args:
            step_type(str):
                Either "training" or 'testing' str for enviornment 
        """
        pass

    def get_name(self):
        """
        Return name of the agent
        """
        return self.name

    def get_reward_function(self):
        """
        Returns the value of the reward function
        """
        return self.reward_function(self.env)

    def get_avail_actions(self):
        """
        Function which reads the avaliable actions avaliable to agent
        """
        return self.env.available_actions

    def set_environment(self, enviornment):
        """
        Funciton which assigns the agent to an enviornment
        
        Args:
            enviornment(Object): 
        """
        self.env = enviornment
    
    def set_reward_func(self, reward_function) -> None:
        """
        Funciton which links a reward function to the agent
        
        Args:
            reward_function(func): A function that computes a reward based on 
                the enviornment's current state 
        """
        self.reward_function = reward_function