import time
import torch

from transformer.loss import cal_ce_loss, cal_ctc_ce_loss
from utils.solver import Solver


class Transformer_Solver(Solver):
    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):
            padded_input, input_lengths, padded_target = data
            padded_input = padded_input.cuda()
            input_lengths = input_lengths.cuda()
            padded_target = padded_target.cuda()
            pred, gold = self.model(padded_input, input_lengths, padded_target)
            ce_loss = cal_ce_loss(
                pred, gold, smoothing=self.label_smoothing)
            loss = ce_loss
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {} | Iter {} | Current Loss {:.3f} | lr {:.3e} | {:.1f} ms/batch | step {}'.
                      format(epoch + 1, i + 1, ce_loss.item(), self.optimizer.optimizer.param_groups[0]["lr"],
                             1000 * (time.time() - start) / (i + 1), self.optimizer.step_num),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)


class Transformer_CTC_Solver(Solver):
    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, data in enumerate(data_loader):
            padded_input, input_lengths, padded_target = data
            padded_input = padded_input.cuda()
            input_lengths = input_lengths.cuda()
            padded_target = padded_target.cuda()
            pred, gold = self.model(padded_input, input_lengths, padded_target)
            ctc_loss, ce_loss = cal_ctc_ce_loss(
                pred, gold, smoothing=self.label_smoothing)
            loss = ctc_loss + ce_loss
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {} | Iter {} | Current Loss {:.3f}|{:.3f} | {:.1f} ms/batch'.
                      format(epoch + 1, i + 1, ctc_loss.item(), ce_loss.item(),
                             1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)
