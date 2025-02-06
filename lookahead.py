import torch
import torch.optim as optim

class Lookahead(optim.Optimizer):
    """Implements Lookahead optimizer.
    
    Lookahead Optimizer: k steps forward, 1 step back
    Paper: https://arxiv.org/abs/1907.08610
    """
    def __init__(self, optimizer, alpha=0.5, k=6):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Invalid alpha {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k {k}")
        self.optimizer = optimizer
        self.alpha = alpha  # update fraction
        self.k = k          # lookahead steps
        self.param_groups = self.optimizer.param_groups

        # Create slow weights
        self.slow_params = []
        for group in self.param_groups:
            group_slow = []
            for p in group['params']:
                p_clone = p.clone().detach()
                group_slow.append(p_clone)
            self.slow_params.append(group_slow)

        self.step_counter = 0

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        # Every k steps, update slow weights
        if self.step_counter % self.k == 0:
            for group_idx, group in enumerate(self.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_params[group_idx][p_idx]
                    slow += self.alpha * (p.data - slow)
                    p.data.copy_(slow)
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            'slow_params': self.slow_params,
            'step_counter': self.step_counter
        }
        fast_state_dict.update(slow_state)
        return fast_state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        slow_state_dict = {
            'slow_params': state_dict['slow_params'],
            'step_counter': state_dict['step_counter']
        }
        fast_state_dict = {key: value for key, value in state_dict.items()
                          if key not in slow_state_dict}
        self.optimizer.load_state_dict(fast_state_dict)
        self.slow_params = slow_state_dict['slow_params']
        self.step_counter = slow_state_dict['step_counter']
