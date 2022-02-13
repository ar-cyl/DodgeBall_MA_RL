import torch as T
import torch.nn.functional as F
from agent import Agent
import csv

grads_f = open("grads_actor.txt", 'a')
writerr = csv.writer(grads_f)

grads_ff = open("grads_critic.txt", 'a')
writerrr = csv.writer(grads_ff)
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 scenario='ctf',  alpha=0.01, beta=0.01, fc1=128, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        assert len(actions) == 4
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards,dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1) #from target_actor(new state)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1) #from actor(state)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1) #from observed action (added with noise)

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.zero_grad()
        
        
        Agent.critic[0].optimizer.zero_grad()
        critic_value_ = Agent.critic[1].forward(states_, new_actions)
        # print(Agent.critic[0].forward(states, old_actions))
        # print(Agent.critic[0].forward(states, old_actions).flatten())
        
        critic_value_[dones] = 0.0
        ####critic_value_ = critic_value_.flatten()
        critic_value = Agent.critic[0].forward(states, old_actions)
        #print(rewards)
        target = rewards + 0.99*critic_value_
        #print(target, critic_value)
        critic_loss = F.mse_loss(target, critic_value)
        #T.save(critic_loss, 'critic_loss.pt')
        critic_loss.backward(retain_graph=True)
        writerrr.writerow(Agent.critic[0].fc4.weight.grad)
        Agent.critic[0].optimizer.step()
        actors_loss = Agent.critic[0].forward(states, mu)
        actors_loss = T.mean(actors_loss, axis=0)
        #T.save(actors_loss, 'actors_loss.pt')
        #print("here")
        #print(actors_loss.shape)
        # print(actors_loss.flatten.shape())
        for agent_idx, agent in enumerate(self.agents):
            actor_loss = -actors_loss[agent_idx]
            actor_loss.backward(retain_graph=True)

        for agent_idx, agent in enumerate(self.agents):
            writerr.writerow(agent.actor.pi1.weight.grad)
            print(agent.actor.f1.weight.data)
            agent.actor.optimizer.step()
            agent.update_network_parameters()
        