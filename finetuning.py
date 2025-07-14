import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
from modeling.pytorch_borzoi_utils import (
    poisson_multinomial_torch,
    add_flashzoi_weight_decay,
)
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
import os
import tqdm
from accelerate import Accelerator

if __name__ == "__main__":
    local_world_size = 2
    accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False)

    # Training params
    run_name = "flashzoi_from_borzoi"
    # Dataset params
    train_iterator_human = iter(training_loader_human)  # TODO: define dataset above
    train_iterator_mouse = iter(training_loader_mouse)  # TODO: define dataset above
    num_steps = 1_000_000  # TODO: calculate based on dataset size and batch size
    # Model params
    device = accelerator.device
    batch_size = 1  # with local_world_size 2, this is effectively batch size 2
    lr = 0.00006
    warmup_steps = 20000
    num_epochs = 100
    eval_every_n = 256
    clip_global_norm = 0.2
    weight_decay = 1e-8
    weight_decay_transformer = 2e-8
    checkpoint_dir = ""  # TODO: set to your path

    loss_fn = poisson_multinomial_torch

    config = BorzoiConfig.from_pretrained("johahi/flashzoi-replicate-0")
    flashzoi = Borzoi.from_pretrained(
        "johahi/borzoi-replicate-0", config=config, ignore_mismatched_sizes=True
    )

    for name, param in flashzoi.named_parameters():
        if "transformer" in name:
            accelerator.print(name, param.requires_grad, param.shape)
            continue
        elif "head" in name and "mouse" in name:
            accelerator.print(name, param.requires_grad)
            continue
        else:
            param.requires_grad = False
            continue

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print("Trainable params: ", count_parameters(flashzoi))

    dropout_modules = [
        module for module in flashzoi.modules() if isinstance(module, torch.nn.Dropout)
    ]
    batchnorm_modules = [
        module
        for module in flashzoi.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    flashzoi.train()

    parameters = add_flashzoi_weight_decay(
        flashzoi,
        accelerator,
        weight_decay=weight_decay,
        weight_decay_transformer=weight_decay_transformer,
    )

    optimizer = torch.optim.AdamW(parameters, lr=lr)

    def warmup(current_step: int):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps)
        else:
            return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    train_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=num_steps - warmup_steps,
        verbose=False,
    )

    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, train_scheduler], [warmup_steps]
    )

    flashzoi = nn.SyncBatchNorm.convert_sync_batchnorm(flashzoi)

    flashzoi, optimizer, scheduler = accelerator.prepare(flashzoi, optimizer, scheduler)

    train_iterator_human = iter(training_loader_human)
    train_iterator_mouse = iter(training_loader_mouse)

    accelerator.init_trackers(
        "flashzoi", init_kwargs={"wandb": {"name": f"{run_name}"}}
    )

    for i in tqdm.tqdm(range(num_steps)):
        optimizer.zero_grad()
        # do step with human
        inputs, targets = next(train_iterator_human)
        with accelerator.autocast():
            outputs = flashzoi(inputs, is_human=True).permute(0, 2, 1)
        targets = targets.to(torch.float32)
        loss = loss_fn(outputs, targets)
        accelerator.log(
            {"loss": loss},
        )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(flashzoi.parameters(), clip_global_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # do step with mouse
        inputs, targets = next(train_iterator_mouse)
        with accelerator.autocast():
            outputs = flashzoi(inputs, is_human=False).permute(0, 2, 1)
        targets = targets.to(torch.float32)
        loss = loss_fn(outputs, targets)
        accelerator.log(
            {"loss": loss},
        )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(flashzoi.parameters(), clip_global_norm)
        optimizer.step()
        scheduler.step()

        # eval on human if neccessary
        accelerator.log({"lr": scheduler.get_last_lr()[-1]})
        if i % eval_every_n == 0 and i != 0:
            # evaluate(accelerator,flashzoi,val_loader_human) # TODO: write custom evaluation function
            flashzoi.train()
        if i % 12000 == 0:
            accelerator.save_state(output_dir=f"{checkpoint_dir}/step_{i}_{run_name}")

    accelerator.end_training()
