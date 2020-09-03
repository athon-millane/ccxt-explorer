import numpy as np

# hyperparameters
ESTIMATOR_WINDOW = 100
REWARD_HORIZON = 100
DISCOUNT_DECAY = 0.95

def run_simulation(x, y, agent, train_agent=False, real_spread=False):

    def current_reward(t, position):
        # custom reward 
        return ((y[t+1] - y[t]) + (x[t] - x[t+1])) * position

    def calculate_expected_reward_TD(t, episodeSARs):
        
        expected_future_pnl = 0.

        # initialise discount factor
        discount_factor = DISCOUNT_DECAY
        
        position = episodeSARs[t][2] - 1
        
        # use bellman update equation over reward horizon of 100
        for tau in range(2, REWARD_HORIZON + 1):
            if t+tau < len(episodeSARs):
                expected_future_pnl += discount_factor * ((y[t+tau] - y[t+tau-1]) \
                                            + (x[t+tau-1] - x[t+tau])) * position
                discount_factor *= DISCOUNT_DECAY # decay discount factor

        if t+tau < len(episodeSARs):
            final_state = [episodeSARs[t+tau][0], episodeSARs[t+tau][1]]
            Q = agent.getMaxQ(final_state) # Get reward of best action from final state

            return expected_future_pnl + DISCOUNT_DECAY * Q
        
        else: # final episode
            return expected_future_pnl

    current_position = 0
    current_pnl = 0
    episodeSARs = []
    
    if real_spread:
        spreads = []

    SARs = []
    for t in range(len(x) - REWARD_HORIZON):

        # calculate P&L accrued from whatever position we were in at last time step.
        if current_position == 1:     # long
            current_pnl += (y[t] - y[t-1]) + (x[t-1] - x[t])
        elif current_position == -1:  # short
            current_pnl += (y[t-1] - y[t]) + (x[t] - x[t-1])
            
        spread = y[t] - x[t]
        if real_spread:
            spreads.append(spread)
            start_idx = t - ESTIMATOR_WINDOW
            if start_idx < 0:
                start_idx = 0

            mu_estimate = np.mean(spreads[start_idx:t+1])
            spread = spread - mu_estimate

        # summarize the current state of things, and query the agent for our next action
        state = [spread, current_position]
        action = agent.act(state)
        
        current_position = action - 1
        
        if train_agent:
            reward = current_reward(t, current_position)
            sar = [state[0], state[1], action, reward]
            episodeSARs.append(sar)
        else:
            # added by athon
            reward = current_reward(t, current_position)
            sar = [state[0], state[1], action, reward]
            SARs.append(sar)

    if train_agent:
        for t in range(len(episodeSARs)):
            expected_future_pnl = calculate_expected_reward_TD(t, episodeSARs)
            reward_label = episodeSARs[t][3] + expected_future_pnl
            tmpSAR = [episodeSARs[t][0], episodeSARs[t][1], episodeSARs[t][2], reward_label]
            agent.remember(tmpSAR)

    return current_pnl, SARs