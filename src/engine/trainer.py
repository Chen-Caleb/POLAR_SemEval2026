import torch
from transformers import Trainer

class FGMTrainer(Trainer):
    """
    自定义 Trainer：支持 FGM (Fast Gradient Method) 对抗训练
    """
    def __init__(self, fgm_epsilon=1.0, fgm_name='word_embeddings', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgm_epsilon = fgm_epsilon
        self.fgm_name = fgm_name
        # 存储扰动值，用于恢复权重
        self.perturbations = {}

    def training_step(self, model, inputs):
        """
        重写训练步骤以注入扰动
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 1. 正常的前向传播和反向传播，获取原始梯度
        loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        # 2. 对抗扰动注入 (FGM)
        # 只有在 epsilon > 0 时才触发，方便通过配置开关
        if self.fgm_epsilon > 0:
            self.fgm_attack(model)
            loss_adv = self.compute_loss(model, inputs)
            if self.args.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            loss_adv.backward() # 反向传播对抗损失的梯度
            self.fgm_restore(model) # 恢复 Embedding 权重

        return loss.detach()

    def fgm_attack(self, model):
        """沿着梯度上升方向注入扰动，并保存扰动值"""
        self.perturbations = {}  # 清空之前的扰动
        for name, param in model.named_parameters():
            if param.requires_grad and self.fgm_name in name and param.grad is not None:
                norm = torch.norm(param.grad)
                if norm != 0:
                    # 计算扰动值
                    r_at = self.fgm_epsilon * param.grad / norm
                    # 保存扰动值用于后续恢复
                    self.perturbations[name] = r_at.clone()
                    # 注入扰动
                    param.data.add_(r_at)

    def fgm_restore(self, model):
        """恢复被扰动的权重：减去之前添加的扰动值"""
        for name, param in model.named_parameters():
            if param.requires_grad and self.fgm_name in name and name in self.perturbations:
                # 减去之前添加的扰动值，恢复原始权重
                param.data.sub_(self.perturbations[name])
        # 清空扰动值，释放内存
        self.perturbations = {}