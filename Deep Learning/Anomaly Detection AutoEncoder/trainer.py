from torch.utils.tensorboard import SummaryWriter
import pathlib
from torchvision import transforms, datasets
import torch
import os
from ignite.engie import Engie, Events
from ignite.metrics import Loss, RunningAverage, ConfusionMatrix
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

def create_summary_writer(model, train_loader, log_dir, save_graph, device):
    writer = SummaryWriter(log_dir=log_dir)

    if save_graph:
        images, labels = next(iter(train_loader))
        images = images.to(device)

        writer.add_graph(model, images)

    return writer

def train(model, optimizer, loss_func, train_loader, val_loader,
          log_dir, device, epochs, log_interval,
          load_weight_path = None, save_graph = False):
    model.to(device)
    if load_weight_path is not None:
        model.load_state_dict(torch.load(load_weight_path))
    
    optimizer = optimizer(model.parameters())
    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, _ = batch
        x = x.to(device)
        y = model(x)
        loss = loss_func(y, x)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_function(engine, batch):
        model.eval()
        x, _ = batch
        x = x.to(device)
        y = model(x)
        loss = loss_func(y, x)
        return loss.item()
    
    trainer = Engie(process_function)
    evaluator = Engie(eval_function)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')

    writer = create_summary_writer(model, train_loader, log_dir, save_graph, device)
    
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss
    
    to_save = {'model': model}
    handler = Checkpoint(to_save, DiskSaver(log_dir, create_dir=True, require_empty=False), score_function=score_function, n_saved=5,
                         file_name_prefix='best', global_step_transform=global_step_from_engine(trainer))
    evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(f'Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}]/'
              f'Loss: {engine.state.metrics["loss"]:.3f}')
        writer.add_scalar("training/loss", engine.state.output,
                          engine.state.iteration)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        print(f'Training Results - Epoch: {engine.state.epoch} Avg loss: {avg_loss:.3f}')
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        print(f'Validation Results - Epoch: {engine.state.epoch} Avg loss: {avg_loss:.3f}')
        writer.add_scalar("valdation/avg_loss", avg_loss, engine.state.epoch)
    
    trainer.run(train_loader, max_epochs=epochs)
    writer.close()
    