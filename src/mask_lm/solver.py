import time
import torch
import os

from mask_lm.loss import cal_ce_mask_loss
from utils.solver import Solver


class Mask_LM_Solver(Solver):
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.epochs = args.epochs
        self.label_smoothing = args.label_smoothing
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self._continue = args._continue
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)

            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.
                  format(epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            file_path = os.path.join(
                self.save_folder, 'epoch-%d.model' % (epoch + 1))
            torch.save(self.model.serialize(self.model,
                                            self.optimizer, epoch + 1,
                                            tr_loss=self.tr_loss,
                                            cv_loss=self.cv_loss),
                       file_path)
            print('Saving checkpoint model to %s' % file_path)

            if epoch > 9:
                file_path = file_path.replace('epoch-%d.model' % (epoch + 1),
                                              'epoch-%d.model' % (epoch - 10))
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.serialize(self.model,
                                                self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, data in enumerate(data_loader):
            padded_input = data
            input_lengths = (data > 0).int().sum(-1)
            padded_input = padded_input.cuda()
            input_lengths = input_lengths.cuda()
            logits_AE, logits, mask = self.model(padded_input, input_lengths)
            ce_loss = cal_ce_mask_loss(
                logits_AE, padded_input, mask, smoothing=self.label_smoothing)
            loss = ce_loss
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {} | Iter {} | batch [{}, {}] | Loss {:.3f} | lr {:.3e} | {:.1f} ms/batch | step {}'.
                      format(epoch + 1, i + 1, padded_input.size(0), padded_input.size(1), ce_loss.item(), self.optimizer.optimizer.param_groups[0]["lr"],
                      1000 * (time.time() - start) / (i + 1), self.optimizer.step_num))

        return total_loss / (i + 1)
