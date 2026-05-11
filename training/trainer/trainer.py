import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard.writer import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT'] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
                                                               
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}                                                                              
        self.logger = logger
        self.metric_scoring = metric_scoring
                                                
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()                     

                          
        self.timenow = time_now
                               
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
                                   
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
                                       
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp']:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True, output_device=self.config['local_rank'])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if not os.path.isfile(model_path):
            raise NotImplementedError(
                f"=> no model found at '{model_path}'")
        saved = torch.load(model_path, map_location='cpu')
        suffix = model_path.split('.')[-1]
        if suffix == 'p':
            self.model.load_state_dict(saved.state_dict())
        else:
            self.model.load_state_dict(saved)
        self.logger.info(f'Model found in {model_path}')

    def _make_save_dir(self, *path_parts: str) -> str:
                                                               
        save_dir = os.path.join(self.log_dir, *path_parts)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = self._make_save_dir(phase, dataset_key)
        save_path = os.path.join(save_dir, "ckpt_best.pth")
        if not self.config['ddp'] and 'svdd' in self.config['model_name']:
            torch.save({'R': self.model.R,
                        'c': self.model.c,
                        'state_dict': self.model.state_dict()}, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self._make_save_dir()
        save_path = os.path.join(save_dir, "swa.pth")
        torch.save(self.swa_model.state_dict(), save_path)                            
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = self._make_save_dir(phase, dataset_key)
        save_path = os.path.join(save_dir, "feat_best.npy")
        np.save(save_path, fea)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = self._make_save_dir(phase, dataset_key)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = self._make_save_dir(phase, dataset_key)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self, data_dict):
        if self.config['optimizer']['type'] == 'sam':
            return self._train_step_sam(data_dict)

        predictions = self.model(data_dict)
        losses = (
            self.model.module.get_losses(data_dict, predictions)                            
            if type(self.model) is DDP
            else self.model.get_losses(data_dict, predictions)                          
        )
        self.optimizer.zero_grad()
        losses['overall'].backward()
        self.optimizer.step()
        return losses, predictions

    def _train_step_sam(self, data_dict):
        for i in range(2):
            predictions = self.model(data_dict)
            losses = self.model.get_losses(data_dict, predictions)                          
            if i == 0:
                pred_first = predictions
                losses_first = losses
            self.optimizer.zero_grad()
            losses['overall'].backward()
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        return losses_first, pred_first


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):
        self.logger.info(f"===> Epoch[{epoch}] start!")
        times_per_epoch = 2 if epoch >= 1 else 1
        test_step = max(1, len(train_data_loader) // times_per_epoch)
        step_cnt = epoch * len(train_data_loader)

        self.save_data_dict('train', train_data_loader.dataset.data_dict, ','.join(self.config['train_dataset']))

        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)
        test_best_metric = None
        use_swa = self.config.get('SWA', False) and epoch > self.config.get('swa_start', float('inf'))

        for iteration, data_dict in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            self.setTrain()
            self._move_data_to_gpu(data_dict)
            losses, predictions = self.train_step(data_dict)

            if use_swa:
                self.swa_model.update_parameters(self.model)                            

            self._record_batch(data_dict, predictions, losses, train_recorder_loss, train_recorder_metric)

            if iteration % 300 == 0 and self.config['local_rank'] == 0:
                self._log_and_clear_recorders(epoch, step_cnt, train_recorder_loss, train_recorder_metric)

            if (step_cnt + 1) % test_step == 0:
                test_best_metric = self._maybe_run_test(epoch, iteration, test_data_loaders, step_cnt)

            step_cnt += 1
        return test_best_metric

    def _record_batch(self, data_dict, predictions, losses, recorder_loss, recorder_metric):
                                                                       
        batch_metrics = (
            self.model.module.get_train_metrics(data_dict, predictions)                            
            if type(self.model) is DDP
            else self.model.get_train_metrics(data_dict, predictions)                          
        )
        for name, value in batch_metrics.items():
            recorder_metric[name].update(value)
        for name, value in losses.items():
            recorder_loss[name].update(value)

    def _log_and_clear_recorders(self, epoch, step_cnt, recorder_loss, recorder_metric):
                                                                                   
        if self.config.get('SWA') and (epoch > self.config.get('swa_start', float('inf')) or self.config.get('dry_run')):
            self.scheduler.step()
        self._log_train_records(step_cnt, recorder_loss, 'loss', 'train_loss')
        self._log_train_records(step_cnt, recorder_metric, 'metric', 'train_metric')
        for recorder in recorder_loss.values():
            recorder.clear()
        for recorder in recorder_metric.values():
            recorder.clear()

    def _move_data_to_gpu(self, data_dict: dict) -> None:
                                                                      
        for key in data_dict:
            if data_dict[key] is not None and key != 'name':
                data_dict[key] = data_dict[key].cuda()

    def _log_train_records(self, step_cnt: int, recorder_dict, kind: str, tb_prefix: str) -> None:
                                                                             
        train_datasets = ','.join(self.config['train_dataset'])
        log_str = f"Iter: {step_cnt}    "
        for k, v in recorder_dict.items():
            v_avg = v.average()
            if v_avg is None:
                log_str += f"training-{kind}, {k}: not calculated    "
                continue
            log_str += f"training-{kind}, {k}: {v_avg}    "
            writer = self.get_writer('train', train_datasets, k)
            writer.add_scalar(f'{tb_prefix}/{k}', v_avg, global_step=step_cnt)
        self.logger.info(log_str)

    def _maybe_run_test(self, epoch, iteration, test_data_loaders, step_cnt):
                                                                          
        should_test = (
            test_data_loaders is not None
            and (not self.config['ddp'] or dist.get_rank() == 0)
        )
        if not should_test:
            return None
        self.logger.info("===> Test start!")
        return self.test_epoch(epoch, iteration, test_data_loaders, step_cnt)

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        real_idx = np.where(label == 0)[0]
        fake_idx = np.where(label == 1)[0]
        acc_real = np.count_nonzero(judge[real_idx]) / len(real_idx)
        acc_fake = np.count_nonzero(judge[fake_idx]) / len(fake_idx)

        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):                                                             
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists = []
        label_lists = []
        for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            data_dict.pop('label_spe', None)                                        
            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)
            self._move_data_to_gpu(data_dict)

            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                losses = (
                    self.model.module.get_losses(data_dict, predictions)                            
                    if type(self.model) is DDP
                    else self.model.get_losses(data_dict, predictions)                          
                )
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(feature_lists)

    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset):                                                             
        current_score = metric_one_dataset[self.metric_scoring]
        best_metric = self.best_metrics_all_time[key].get(
            self.metric_scoring,
            float('inf') if self.metric_scoring == 'eer' else float('-inf'),
        )
        improved = current_score < best_metric if self.metric_scoring == 'eer' else current_score > best_metric
        if improved:
            self.best_metrics_all_time[key][self.metric_scoring] = current_score
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        self._log_test_losses(step, key, losses_one_dataset_recorder)
        self._log_test_metrics(step, key, metric_one_dataset)

    def _log_test_losses(self, step, key, losses_recorder):
                                             
        if losses_recorder is None:
            return
        loss_str = f"dataset: {key}    step: {step}    "
        for k, v in losses_recorder.items():
            writer = self.get_writer('test', key, k)
            v_avg = v.average()
            if v_avg is None:
                print(f'{k} is not calculated')
                continue
            writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
            loss_str += f"testing-loss, {k}: {v_avg}    "
        self.logger.info(loss_str)

    def _log_test_metrics(self, step, key, metric_one_dataset):
                                              
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k in ('pred', 'label', 'dataset_dict'):
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar('test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar('test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)
    def test_epoch(self, epoch, iteration, test_data_loaders, step):                                                             
        self.setEval()

        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}

        for key in test_data_loaders:
            self.save_data_dict('test', test_data_loaders[key].dataset.data_dict, key)

            losses_recorder, predictions_nps, label_nps, _ = self.test_one_dataset(test_data_loaders[key])
            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps, y_true=label_nps,
                img_names=test_data_loaders[key].dataset.data_dict['image'],
            )
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name] += value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]

            if type(self.model) is AveragedModel:
                metric_str = "Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch, iteration, step, losses_recorder, key, metric_one_dataset)

        num_datasets = len(test_data_loaders)
        if num_datasets > 0 and self.config.get('save_avg', False):
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= num_datasets
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time

    @torch.no_grad()
    def inference(self, data_dict):
        return self.model(data_dict, inference=True)
