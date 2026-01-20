import torch
from transformers import Trainer

class FGMTrainer(Trainer):
    """
    Custom Trainer with FGM (Fast Gradient Method) adversarial training support.
    """
    def __init__(self, fgm_epsilon=1.0, fgm_name='word_embeddings', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgm_epsilon = fgm_epsilon
        self.fgm_name = fgm_name
        # Store perturbations for later restoration
        self.perturbations = {}

    def training_step(self, model, inputs):
        """
        Override training step to inject adversarial perturbations.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 1. Standard forward + backward to obtain original gradients
        loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        # 2. Adversarial perturbation injection (FGM)
        # Only enabled when epsilon > 0 so it can be toggled via config.
        if self.fgm_epsilon > 0:
            self.fgm_attack(model)
            loss_adv = self.compute_loss(model, inputs)
            if self.args.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            loss_adv.backward()  # Backpropagate adversarial loss
            self.fgm_restore(model)  # Restore embedding weights

        return loss.detach()

    def fgm_attack(self, model):
        """Inject perturbations along the gradient direction and save them."""
        self.perturbations = {}  # Clear previous perturbations
        for name, param in model.named_parameters():
            if param.requires_grad and self.fgm_name in name and param.grad is not None:
                norm = torch.norm(param.grad)
                if norm != 0:
                    # Compute perturbation
                    r_at = self.fgm_epsilon * param.grad / norm
                    # Save perturbation for later restoration
                    self.perturbations[name] = r_at.clone()
                    # Inject perturbation
                    param.data.add_(r_at)

    def fgm_restore(self, model):
        """Restore perturbed weights by subtracting previously added perturbations."""
        for name, param in model.named_parameters():
            if param.requires_grad and self.fgm_name in name and name in self.perturbations:
                # Subtract perturbation to restore original weights
                param.data.sub_(self.perturbations[name])
        # Clear stored perturbations
        self.perturbations = {}