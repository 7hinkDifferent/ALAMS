import numpy as np
from .strategy import Strategy
from .least_confidence import LeastConfidence
from .kmeans_sampling import KMeansSampling

# MAB
# selects strategy based on expected scores (UCB)
# uses selected strategy to query samples

# cold-start
# query equal number of samples from each strategy in the beginning
# then use UCB to select strategy based on historical performance

# C as exploration parameter, approx 30 % for exploration
# need to track the balance between exploration and exploitation
# pi = expected_score + C * sqrt( (2 * log(total_count)) / count )

# when using MAB, change round and query number accordingly
# because we must feedback after each round

class MultiArmBandit(Strategy):
    def __init__(self, dataset, net):
        super(MultiArmBandit, self).__init__(dataset, net)
        self.strategies = [
            LeastConfidence(dataset, net),
            KMeansSampling(dataset, net),
        ]  # list of strategies (arms)å
        self.total_count = 0  # total number of queries made
        self.records = []
        for _ in self.strategies:
            self.records.append({
                "scores": 0.0,
                "count": 0,
                "current_reward": 0.0,
                "total_reward": 0.0,
                "metrics": [],
            })

        self.last_selected_strategy = None
        self.last_metrics = None
        self.decay = 1.0 # 0.9  # decay factor for expected score
        self.C = 1.0  # exploration parameter
        
        
    def feedback(self, info):
        # define reward as relative improvement in training accuracy
        if self.last_selected_strategy is not None:
            strategy_idx = self.last_selected_strategy
            if self.last_metrics is not None:
                current_val_loss = info["best_val_loss"]
                reward = (self.last_metrics - current_val_loss) / self.last_metrics  # relative decrease in val loss
                # update records
                self.records[strategy_idx]["current_reward"] = reward
                self.records[strategy_idx]["metrics"].append(current_val_loss)
                self.records[strategy_idx]["total_reward"] = reward + self.decay * self.records[strategy_idx]["total_reward"]
                # update expected score, exploration term will be added during query
                self.records[strategy_idx]["scores"] = self.records[strategy_idx]["total_reward"] / (self.records[strategy_idx]["count"])
            
        self.last_metrics = info["best_val_loss"]

        return self.records

    def query(self, n):
        info = {
            "selected_strategies": 0,
            "strategy_scores": 0.0,
        }
        # cold-start: query equal number of samples from each strategy
        for i in range(len(self.strategies)):
            if self.records[i]["count"] == 0:
                q_idxs, meta = self.strategies[i].query(n)
                self.records[i]["count"] += 1
                self.total_count += 1
                self.last_selected_strategy = i
                return q_idxs, info.update({
                    "selected_strategies": [i],
                    "strategy_scores": [0.0],
                    **meta
                })
        
        # select strategy based on UCB
        ucb_values = []
        exploitation_values = []
        exploration_values = []
        for i in range(len(self.strategies)):
            exploitation_value = self.records[i]["scores"]
            exploration_value = np.sqrt((2 * np.log(self.total_count)) / (self.records[i]["count"]))
            ucb = exploitation_value + self.C * exploration_value
            ucb_values.append(ucb)
            exploitation_values.append(exploitation_value)
            exploration_values.append(exploration_value)
        ucb_values = np.array(ucb_values)
        selected_strategy = ucb_values.argmax()
        q_idxs, meta = self.strategies[selected_strategy].query(n)
        # update records
        self.records[selected_strategy]["count"] += 1
        self.total_count += 1
        self.last_selected_strategy = selected_strategy
        info.update({
            "selected_strategies": [selected_strategy],
            "strategy_scores": [ucb_values[selected_strategy]],
            "all_scores": ucb_values.tolist(),
            "exploitation_values": exploitation_values,
            "exploration_values": exploration_values,
            **meta
        })

        return np.array(q_idxs), info