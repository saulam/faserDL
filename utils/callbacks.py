import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning


class CustomFinetuningReversed(pl.callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=5, gradual_unfreeze_steps=5, lr_factor=0.1):
        """
        unfreeze_at_epoch: Epoch at which to start unfreezing the classifier.
        gradual_unfreeze_steps: Number of steps to unfreeze the remaining layers gradually.
        lr_factor: Factor to reduce learning rate for newly unfrozen layers.
        """
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.gradual_unfreeze_steps = gradual_unfreeze_steps
        self.lr_factor = lr_factor
        self.unfrozen_layers = set()  # Track which layers have already been unfrozen
        self.total_layers_unfrozen = 0  # Track the number of total layers unfrozen

    def freeze_before_training(self, pl_module):
        """
        Initial freezing before training starts.
        """
        # Freeze everything except cls_layer
        if pl_module.model.finetuning:
            # Freeze the entire model
            self.freeze(pl_module.model, train_bn=False)
            print("Freezing entire model")

            # Unfreeze the cls_layer immediately, so it's trainable from the start
            if hasattr(pl_module.model, 'cls_layer'):
                self.make_trainable(pl_module.model.cls_layer)
                self.unfrozen_layers.add(pl_module.model.cls_layer)
                print(f"Unfreezing layer: cls_layer with learning rate: base learning rate (full LR)")

    def finetune_function(self, pl_module, epoch, optimizer, opt_idx):
        """
        Unfreezing logic for progressive finetuning.
        Unfreezes the classification layer first, then the decoder layers, followed by the encoder layers.
        """
        base_lr = optimizer.param_groups[0]['lr']  # Get the base learning rate from the optimizer

        # Gradually unfreeze the decoder first, then the encoder
        if epoch >= self.unfreeze_at_epoch:
            # Only unfreeze a few layers per step, determined by gradual_unfreeze_steps
            unfreeze_limit = self.total_layers_unfrozen + self.gradual_unfreeze_steps
            decoder_layers = [(x, y) for x, y in zip(pl_module.model.upsample_layers[::-1], pl_module.model.stages_dec[::-1])]
            encoder_layers = [(x, y) for x, y in zip(pl_module.model.downsample_layers[::-1], pl_module.model.stages_enc[::-1])]

            # Unfreeze decoder layers first, and then move to encoder layers
            if self.total_layers_unfrozen < len(decoder_layers):
                self._unfreeze_layers_step(pl_module, optimizer, decoder_layers, base_lr, unfreeze_limit)
            else:
                self._unfreeze_layers_step(pl_module, optimizer, encoder_layers, base_lr, unfreeze_limit)

    def _get_module_name(self, pl_module, layer):
        """
        Helper function to get the module name from the model.
        """
        for name, module in pl_module.model.named_modules():
            if module is layer:
                return name
        return "Unknown"

    def _unfreeze_layers_step(self, pl_module, optimizer, layer_pairs, base_lr, unfreeze_limit):
        """
        Unfreeze the layers gradually without repeating the already unfrozen layers.
        This function ensures that only new layers are unfrozen at each step.
        """
        for i, (layer1, layer2) in enumerate(layer_pairs):
            # Unfreeze both layers in the pair only if they haven't been unfrozen yet and within the current unfreeze limit
            if layer1 not in self.unfrozen_layers and layer2 not in self.unfrozen_layers and self.total_layers_unfrozen < unfreeze_limit:
                # Unfreeze both layers in the pair
                lr_for_new_layers = base_lr * self.lr_factor
                self.unfreeze_and_add_param_group(layer1, optimizer, lr=lr_for_new_layers, train_bn=True)
                self.unfreeze_and_add_param_group(layer2, optimizer, lr=lr_for_new_layers, train_bn=True)
                # Add both layers to the unfrozen set
                self.unfrozen_layers.add(layer1)
                self.unfrozen_layers.add(layer2)
                self.total_layers_unfrozen += 2
                layer1_name = self._get_module_name(pl_module, layer1)
                layer2_name = self._get_module_name(pl_module, layer2)
                # Print layer unfreezing information for both layers, including learning rate
                print(f"Unfreezing pair: {layer1_name} and {layer2_name}, Index: {i}, Learning rate: {lr_for_new_layers}")
