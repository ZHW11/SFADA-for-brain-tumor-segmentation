from torch.optim.lr_scheduler import LambdaLR

class PolyScheduler(LambdaLR):
    def __init__(self, optimizer, t_total, exponent=0.9, step_interval=1, last_epoch=-1):
        self.t_total = t_total
        self.exponent = exponent
        self.step_interval = step_interval  # 新增参数，默认100
        super(PolyScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        # 每 step_interval 才更新一次，相当于把 step 离散化
        effective_step = step // self.step_interval
        effective_total = self.t_total // self.step_interval
        return (1 - effective_step / effective_total) ** self.exponent