""" Training augmented model """
import tqdm
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import utils
from models.augment_cnn import AugmentCNN

from template_lib.trainer.base_trainer import summary_dict2txtfig

# config = AugmentConfig()
#
# device = torch.device("cuda")
#
# # tensorboard
# writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
# writer.add_text('config', config.as_markdown(), 0)
#
# logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
# config.print_params(logger.info)


def main(config, logger, writer, device, myargs):
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True)

    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype)
    model = nn.DataParallel(model, device_ids=config.gpus).to(device)

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_top1 = 0.
    # training loop
    for epoch in tqdm.tqdm(range(config.epochs), desc="main", file=myargs.stdout):
        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.module.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch,
              config=config, writer=writer, logger=logger, device=device, myargs=myargs)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step,
                        config=config, writer=writer, logger=logger, device=device, myargs=myargs)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


def train(train_loader, model, optimizer, criterion, epoch,
          config, writer, logger, device, myargs):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("Epoch {} LR {}".format(epoch, cur_lr))
    # writer.add_scalar('train/lr', cur_lr, cur_step)

    model.train()
    pbar = tqdm.tqdm(train_loader, desc=f"train {myargs.args.time_str_suffix}", file=myargs.stdout)
    for step, (X, y) in enumerate(pbar):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            pbar.write(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5), file=myargs.stdout)

        # writer.add_scalar('train/loss', loss.item(), cur_step)
        # writer.add_scalar('train/top1', prec1.item(), cur_step)
        # writer.add_scalar('train/top5', prec5.item(), cur_step)
        summary_dict = dict(loss=loss.item(), top1=prec1.item(), top5=prec5.item(), lr=cur_lr)
        summary_dict2txtfig(dict_data=summary_dict, prefix="train", step=cur_step,
                            textlogger=myargs.textlogger)
        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step,
             config, writer, logger, device, myargs):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        pbar = tqdm.tqdm(valid_loader, desc='validate', file=myargs.stdout)
        for step, (X, y) in enumerate(pbar):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                pbar.write(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5), file=myargs.stdout)

    # writer.add_scalar('val/loss', losses.avg, cur_step)
    # writer.add_scalar('val/top1', top1.avg, cur_step)
    # writer.add_scalar('val/top5', top5.avg, cur_step)
    summary_dict = dict(loss=losses.avg, top1=top1.avg, top5=top5.avg)
    summary_dict2txtfig(dict_data=summary_dict, prefix="valid", step=cur_step,
                        textlogger=myargs.textlogger)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return top1.avg


def run(argv_str=None):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if hasattr(myargs.config, 'datasets'):
    prepare_dataset(myargs.config.datasets, cfg=myargs.config)

  config = AugmentConfig(args_list=['--name', myargs.config.args.name,
                                   '--dataset', myargs.config.args.dataset,
                                   '--genotype', myargs.config.args.genotype])

  myargs.config.args.pop('genotype')
  config = config2args(myargs.config.args, config)
  config.path = os.path.join(args1.outdir, 'ptdarts')
  os.makedirs(config.path, exist_ok=True)
  config.plot_path = os.path.join(config.path, 'plot')
  device = torch.device("cuda")

  # writer = SummaryWriter(logdir=os.path.join(config.path, "tb"))
  writer = myargs.writer
  writer.add_text('ptdarts_config', config.as_markdown(), 0)

  # logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
  logger = myargs.logger
  config.print_params(logger.info)

  main(config=config, writer=writer, logger=logger, device=device, myargs=myargs)


if __name__ == '__main__':
  run()
  from template_lib.examples import test_bash

  test_bash.TestingUnit().test_resnet(gpu=os.environ['CUDA_VISIBLE_DEVICES'])



