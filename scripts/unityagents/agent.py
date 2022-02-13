import torch as T
from network import ActorNetwork, CriticNetwork
import csv



class Agent:
    critic = [CriticNetwork(0.01, 2016, 
                            0, 0, n_agents=4, n_actions=5, 
                            chkpt_dir='tmp/maddpg/', name='central_critic'),
                CriticNetwork(0.01, 2016, 
                                            0, 0, n_agents=4, n_actions=5,
                                            chkpt_dir='tmp/maddpg/',
                                            name='central_target_critic')] #[local, target]

    
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=128, 
                    fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
      

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.randn(self.n_actions).to(self.actor.device)
        noise[3:5] = 0 #WARNING: hardcoded
        action = actions + noise

        return actions.detach().cpu().numpy()[0] 

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.critic[1].named_parameters()
        critic_params = self.critic[0].named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.critic[1].load_state_dict(critic_state_dict)




    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic[0].save_checkpoint()
        self.critic[1].save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic[0].load_checkpoint()
        self.critic[1].load_checkpoint()