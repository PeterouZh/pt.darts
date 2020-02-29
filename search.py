""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
import tqdm
from tensorboardX import SummaryWriter

from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot

from template_lib.trainer.base_trainer import summary_dict2txtfig

# config = SearchConfig()

# device = torch.device("cuda")

# tensorboard
# writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
# writer.add_text('config', config.as_markdown(), 0)
#
# logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
# config.print_params(logger.info)


def main(config, writer, logger, device, myargs):
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in tqdm.tqdm(range(config.epochs), desc='main epoch', file=myargs.stdout):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch,
              config=config, writer=writer, logger=logger, device=device, myargs=myargs)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step,
                        config=config, writer=writer, logger=logger, device=device, myargs=myargs)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))
        myargs.textlogger.logstr(itr=epoch+1, genotype=str(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        # plot(genotype.normal, plot_path + "-normal", caption)
        # plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch,
          config, writer, logger, device, myargs):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    pbar = tqdm.tqdm(zip(train_loader, valid_loader), desc='searching', file=myargs.stdout,
                     total=len(train_loader))
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(pbar):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            pbar.write(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    top1=top1, top5=top5), file=myargs.stdout)

        # writer.add_scalar('train/loss', loss.item(), cur_step)
        # writer.add_scalar('train/top1', prec1.item(), cur_step)
        # writer.add_scalar('train/top5', prec5.item(), cur_step)
        # writer.add_scalar('train/grad_norm', grad_norm, cur_step)
        summary_dict = dict(loss=loss.item(), top1=prec1.item(), top5=prec5.item(), grad_norm=grad_norm,
                            lr=lr)
        summary_dict2txtfig(dict_data=summary_dict, prefix="train", step=cur_step,
                            textlogger=myargs.textlogger)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step,
             config, writer, logger, device, myargs):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        pbar = tqdm.tqdm(valid_loader, desc='validating', file=myargs.stdout)
        for step, (X, y) in enumerate(pbar):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                pbar.write(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5), file=myargs.stdout)

    # writer.add_scalar('val/loss', losses.avg, cur_step)
    # writer.add_scalar('val/top1', top1.avg, cur_step)
    # writer.add_scalar('val/top5', top5.avg, cur_step)
    summary_dict = dict(loss=losses.avg, top1=top1.avg, top5=top5.avg)
    summary_dict2txtfig(dict_data=summary_dict, prefix="valid", step=cur_step,
                        textlogger=myargs.textlogger)

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

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

  config = SearchConfig(args_list=['--name', myargs.config.args.name,
                                   '--dataset', myargs.config.args.dataset])
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

