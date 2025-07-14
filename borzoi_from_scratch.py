import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from modeling.borzoi_model import Borzoi
import tqdm
from modeling.pytorch_borzoi_utils import poisson_multinomial_torch
from accelerate import Accelerator

if __name__ == "__main__":

    local_world_size = 2
    accelerator = Accelerator(log_with="wandb", step_scheduler_with_optimizer=False)

    # Training params
    run_name = "borzoi_from_scratch"
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
    checkpoint_dir = ""  # TODO: set to your path

    borzoi = Borzoi(
        flash_attention=False, return_center_bins_only=True, enable_mouse_head=True
    )

    optimizer = torch.optim.Adam(borzoi.parameters(), lr=lr, eps=1e-7)

    def warmup(current_step: int):
        if current_step < warmup_steps:
            return float(current_step / warmup_steps)
        else:
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    loss_fn = poisson_multinomial_torch

    borzoi = nn.SyncBatchNorm.convert_sync_batchnorm(borzoi)

    borzoi, optimizer, scheduler = accelerator.prepare(borzoi, optimizer, scheduler)

    accelerator.init_trackers("borzoi", init_kwargs={"wandb": {"name": f"{run_name}"}})

    for i in tqdm.tqdm(range(num_steps)):
        optimizer.zero_grad()
        # do step with human
        inputs, targets = next(train_iterator_human)
        with accelerator.autocast():
            outputs = borzoi(inputs, is_human=True).permute(0, 2, 1)
        targets = targets.to(torch.float32)
        loss = loss_fn(outputs, targets)
        # the following can probably be replaced with setting the weight decay in the optimizer, but that's how I started
        l2_reg = torch.tensor(0.0, requires_grad=True, device=device)
        for name, param in borzoi.named_parameters():
            if "transformer" in name and "weight" in name and len(param.shape) == 2:
                l2_reg = l2_reg + (param**2).sum() * 1e-8
            elif "weight" in name and len(param.shape) > 2:
                l2_reg = l2_reg + (param**2).sum() * 2e-8
        loss = l2_reg + loss
        accelerator.log(
            {"loss": loss},
        )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(borzoi.parameters(), clip_global_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # do step with mouse
        inputs, targets = next(train_iterator_mouse)
        with accelerator.autocast():
            outputs = borzoi(inputs, is_human=False).permute(0, 2, 1)
        targets = targets.to(torch.float32)
        loss = loss_fn(outputs, targets)
        l2_reg = torch.tensor(0.0, requires_grad=True, device=device)
        for name, param in borzoi.named_parameters():
            if "transformer" in name and "weight" in name and len(param.shape) == 2:
                l2_reg = l2_reg + (param**2).sum() * 1e-8
            elif "weight" in name and len(param.shape) > 2:
                l2_reg = l2_reg + (param**2).sum() * 2e-8
        loss = l2_reg + loss
        accelerator.log(
            {"loss": loss},
        )
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(borzoi.parameters(), clip_global_norm)
        optimizer.step()
        scheduler.step()

        # eval on human if neccessary
        accelerator.log({"lr": scheduler.get_last_lr()[-1]})
        if i % eval_every_n == 0 and i != 0:
            # evaluate(accelerator,borzoi,val_loader_human) # TODO: write custom evaluation function
            borzoi.train()
        if i % 12000 == 0:
            accelerator.save_state(output_dir=f"{checkpoint_dir}/step_{i}_{run_name}")

    accelerator.end_training()
