import numpy as np
import os

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps, max_epochs,
                warmup_learning_rate, hold_base_rate_steps, batch_size, decay_rate, work_dir, global_step):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = global_step
        self.max_epochs = max_epochs
        self.warmup_learning_rate = warmup_learning_rate
        self.hold_base_rate_steps = hold_base_rate_steps
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.work_dir =work_dir



    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def decay_with_warmup(self):


        global_step = self.n_steps
        learning_rate_base = self.init_lr
        total_steps = 40091 * self.max_epochs/self.batch_size
        warmup_learning_rate = self.warmup_learning_rate
        warmup_steps=self.n_warmup_steps
        hold_base_rate_steps = self.hold_base_rate_steps
        decay_rate = self.decay_rate
        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                             'warmup_steps.')
        learning_rate = learning_rate_base * decay_rate ** (global_step - warmup_steps - hold_base_rate_steps)
        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                     learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                         learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)





    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.decay_with_warmup()


        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            if self.n_steps % 20 == 0:
                self.print_log("Step: {} lr:{}".format(self.n_steps, lr))

    def print_log(self,str):
        print(str)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        with open('{}/log.txt'.format(self.work_dir), 'a') as f:
            print(str, file=f)