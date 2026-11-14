# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import torch.nn as nn
import os
import numpy as np
import h5py
import copy
import time
import random
import json
from utils.data_utils import read_client_data, results_root
from utils.dlg import DLG
from scipy.optimize import curve_fit


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.device_id = args.device_id
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.model_path = os.path.join(
            results_root, "models", self.dataset, args.model_name,
            f"{self.algorithm}_{self.goal}_{times}_server.pt",
        )
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_test_loss = []
        self.eval_rounds = []
        self.round_selected_count = []
        self.round_privacy_cost = []
        self.round_energy_cost = []
        self.round_model_cost = []
        self.round_total_cost = []
        self.total_train_cost = []
        self.decision_history = []
        self.mechanism_history = []
        self.convergence_round = None

        self.center_cs = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.target_acc = args.target_accuracy
        self.target_acc_count = 0

        for layer in self.global_model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.args, self.dataset, i, is_train=True)
            test_data = read_client_data(self.args, self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            client.target_acc = self.args.target_accuracy
            self.clients.append(client)
            client.model.cpu()
        self.total_data_samples = sum(c.train_samples + c.test_samples for c in self.clients)
        if self.clients and hasattr(self.clients[0], 'client_c'):
            # Adap-CTA follows the original protocol: cc_i^0=g_i(w0), while sc^0=0.
            self.global_c = [torch.zeros_like(p) for p in self.global_model.parameters()]
            if self.algorithm != 'Adap-CTA':
                with torch.no_grad():
                    for client in self.clients:
                        for server_control, client_control in zip(self.global_c, client.client_c):
                            server_control.add_(client_control.to(server_control.device), alpha=1/self.num_clients)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients

            # selected_clients = self.clients

        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        for client in selected_clients:
            if hasattr(client.optimizer, 'total_layer'):
                client.optimizer.round_diagnostics = None

        # start = 0
        # end = 100
        # num_count = 10  # 因为包括0和9，所以数量是10
        # # 生成不重复的随机数字列表
        # random_numbers = random.sample(range(start, end), 10)
        # step = 0
        # for client in selected_clients:
        #     client.id = random_numbers[step]
        #     step += 1
        # selected_clients = self.clients
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        active_clients = self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / len(self.selected_clients)


    def save_global_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.global_model, self.model_path)

    def load_model(self):
        assert os.path.exists(self.model_path)
        self.global_model = torch.load(self.model_path)

    def restore_training_state(self, checkpoint_path, curves_path):
        from flcore.servers.server_privacy import AdapLDPFL
        AdapLDPFL.restore_training_state(self, checkpoint_path, curves_path)

    def model_exists(self):
        return os.path.exists(self.model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.args.model_name + "_" + self.algorithm
        result_path = os.path.join(results_root, "runs") + os.sep
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            # algo = algo + "_" + "ours" + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path + '.partial', 'w') as hf:
                metadata = dict(algorithm=self.algorithm, dataset=self.dataset, model=self.args.model_name,
                    distribution=str(self.args.NIID), total_layer=self.args.total_layer,
                    rounds_requested=self.global_rounds, rounds_completed=len(self.rs_test_acc),
                    num_clients=self.num_clients, join_ratio=self.join_ratio, batch_size=self.batch_size,
                    local_epochs=self.local_epochs, local_learning_rate=self.learning_rate,
                    seed=getattr(self.args, 'seed', 0), server_learning_rate=self.args.server_learning_rate,
                    privacy_protocol='exp(-correction_ratio), a,b removed; normalized cost proxy',
                    evaluation_protocol='global_model_all_test_examples_after_aggregation',
                    batchnorm_evaluation='batch statistics; dropout disabled',
                    cc_initialization='g_i(w0)' if self.algorithm in ('Adap-CTA','FedPVR') else 'not_applicable',
                    sc_initialization=('zero' if self.algorithm == 'Adap-CTA' else 'mean_initial_client_controls' if self.algorithm == 'FedPVR'
                                       else 'zero') if hasattr(self, 'global_c') else 'not_applicable',
                    control_delta_denominator='num_clients' if hasattr(self, 'global_c') else 'not_applicable',
                    control_protocol='original_v1' if self.algorithm == 'Adap-CTA' else 'consistent_v2' if self.algorithm == 'FedPVR' else 'inherited',
                    fixed_decision_l=getattr(self.args, 'fixed_decision_l', None),
                    control_warmup_rounds=getattr(self.args, 'adapcta_control_warmup', 0),
                    source_sha256=os.environ.get('FL_SOURCE_SHA256', 'local'),
                    data_sha256=os.environ.get('FL_DATA_SHA256', 'not_recorded'),
                    cost_timing='post_aggregation', noise_multiplier=self.args.noise_multiplier,
                    target_accuracy=self.target_acc)
                for key, value in metadata.items():
                    if value is not None:
                        hf.attrs[key] = value
                hf.attrs['model_architecture'] = str(self.global_model)
                from utils.data_utils import data_root
                distribution_dir = f'Non-IID-{self.args.NIID}' if self.args.NIID <= 1 else 'IID'
                config_path = os.path.join(data_root, self.dataset, distribution_dir, 'config.json')
                if os.path.isfile(config_path):
                    with open(config_path, encoding='utf-8') as stream:
                        hf.attrs['partition_metadata'] = stream.read()
                for key, value in getattr(self, 'baseline_metadata', {}).items():
                    hf.attrs['baseline_' + key] = value
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                hf.create_dataset('eval_rounds', data=self.eval_rounds)
                hf.create_dataset('round_selected_count', data=self.round_selected_count)
                if self.decision_history:
                    hf.create_dataset('decision_history', data=self.decision_history)
                if self.mechanism_history:
                    for index, name in enumerate(('raw_update_norm', 'corrected_update_norm', 'cc_norm',
                                                  'sc_norm', 'sc_cc_norm', 'correction_magnitude')):
                        hf.create_dataset('round_' + name, data=np.asarray(self.mechanism_history)[:, index])
                    hf.attrs['mechanism_diagnostics'] = 'mean across selected clients, first local SGD step'
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if hasattr(self, 'convergence_round'):
                    hf.create_dataset('convergence_round', data=-1 if self.convergence_round is None else self.convergence_round)
                for name in ('total_train_cost', 'round_total_cost', 'round_privacy_cost',
                             'round_energy_cost', 'round_model_cost'):
                    if hasattr(self, name):
                        hf.create_dataset(name, data=np.asarray(getattr(self, name), dtype=np.float64))
                # hf.create_dataset('total_train_cost', data=self.total_train_cost)
                for c in self.clients:
                    hf.create_dataset("Client_{}_global_acc".format(c.id), data=c.global_acc_history)
                    hf.create_dataset("Client_{}_local_acc".format(c.id), data=c.local_acc_history)
                    hf.create_dataset("Client_{}_train_cost_history".format(c.id), data=c.train_cost)
                    hf.create_dataset("Client_{}_modelAcc_history".format(c.id), data=c.modelAcc_history)
                    hf.create_dataset("Client_{}_privacy_history".format(c.id), data=c.privacy_history)
                    hf.create_dataset("Client_{}_trained_cost".format(c.id), data=c.trained_cost)
                    hf.create_dataset("Client_{}_predict_cost_history".format(c.id), data=c.predict_cost_history)
                    hf.create_dataset("Client_{}_energy_cost_history".format(c.id), data=c.energy_cost_history)

            os.replace(file_path + '.partial', file_path)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            # tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def test_metrics_with_batch_stats(self):
        """Use batch-stat BN evaluation without stochastic Dropout."""
        total_correct = total_samples = 0
        for client in self.clients:
            model = copy.deepcopy(self.global_model).to(self.device)
            model.train()
            # Keep the original batch-stat behavior for BatchNorm, but never
            # enable Dropout during evaluation (it makes the last test round
            # randomly collapse on VGG/Transformer models).
            for module in model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                       nn.Dropout3d, nn.AlphaDropout)):
                    module.eval()
            with torch.no_grad():
                for x, y in client.load_test_data():
                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                        batch_size = x[0].shape[0]
                    else:
                        x = x.to(self.device)
                        batch_size = x.shape[0]
                    y = y.to(self.device)
                    if batch_size == 1:
                        model.eval()
                    output = model(x)
                    if batch_size == 1:
                        model.train()
                        for module in model.modules():
                            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                                   nn.Dropout3d, nn.AlphaDropout)):
                                module.eval()
                    total_correct += (output.argmax(dim=1) == y).sum().item()
                    total_samples += y.shape[0]
        return total_correct, total_samples

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.selected_clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    # def evaluate(self, acc=None, loss=None):
    #
    #
    #     total_samples = 0
    #     total_correct = 0
    #     for client in self.clients:
    #         model = copy.deepcopy(self.global_model)
    #         model.to(self.device)
    #         testloaderfull = client.load_test_data()
    #
    #         test_acc = 0
    #         test_num = 0
    #         y_prob = []
    #         y_true = []
    #
    #         for x, y in testloaderfull:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = model(x)
    #
    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             test_num += y.shape[0]
    #
    #         total_samples += test_num
    #         total_correct += test_acc
    #
    #     test_acc = total_correct / total_samples * 1.0
    #     train_loss = 0
    #
    #     if acc == None:
    #         self.rs_test_acc.append(test_acc)
    #         NIID = 'NIID-' + str(self.args.NIID)
    #         self.save_instant_result(self.rs_test_acc, NIID)
    #     else:
    #         acc.append(test_acc)
    #
    #     if loss == None:
    #         self.rs_train_loss.append(train_loss)
    #     else:
    #         loss.append(train_loss)
    #
    #     print("Averaged Test Accurancy: {:.4f}".format(test_acc))

    def evaluate(self, acc=None, loss=None):
        # Aggregation and control updates are finished; retaining gradients and
        # per-client global snapshots copies several GB back to host RAM.
        for client in self.selected_clients:
            client.optimizer.zero_grad(set_to_none=True)
            if hasattr(client, 'global_c'):
                client.global_c = None
            if hasattr(client, 'global_model'):
                client.global_model = None
        # One global model, all test examples, after this round's aggregation.
        model = copy.deepcopy(self.global_model).to(self.device)
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train()
        correct = samples = 0
        loss_sum = 0.0
        with torch.no_grad():
            for client in self.clients:
                for x, y in client.load_test_data():
                    x = [value.to(self.device) for value in x] if isinstance(x, (tuple, list)) else x.to(self.device)
                    y = y.to(self.device)
                    if len(y) == 1:
                        model.eval()
                    output = model(x)
                    if not torch.isfinite(output).all():
                        raise FloatingPointError('non-finite aggregated model output')
                    loss_sum += torch.nn.functional.cross_entropy(output, y, reduction='sum').item()
                    correct += (output.argmax(1) == y).sum().item()
                    samples += len(y)
                    for module in model.modules():
                        if isinstance(module, nn.modules.batchnorm._BatchNorm):
                            module.train()
        if not samples:
            raise RuntimeError('no test samples')
        accuracy, test_loss = correct / samples, loss_sum / samples
        if acc is not None:
            acc.append(accuracy)
            return
        self.rs_test_acc.append(accuracy)
        self.rs_test_loss.append(test_loss)
        round_index = getattr(self, 'current_round', len(self.rs_test_acc) - 1)
        self.eval_rounds.append(round_index)
        self.round_selected_count.append(len(self.selected_clients))
        sizes = [p.numel() for p in self.global_model.parameters()]
        privacy = energy = model_cost = 0.0
        decisions = []
        for client in self.selected_clients:
            decision = len(sizes) if self.algorithm == 'SCAFFOLD' else int(getattr(client, 'decision_l', 0))
            ratio = sum(sizes[-decision:]) / sum(sizes) if decision else 0.0
            batches = max(1, client.train_samples // self.batch_size)
            privacy += float(np.exp(-ratio))
            energy += 0.82 * batches / (self.total_data_samples / self.num_clients / self.batch_size) + 0.09 + 0.09 * ratio
            model_cost += max(0.0, client.target_acc - accuracy)
            decisions.append(decision)
        total = privacy + energy + model_cost
        self.round_privacy_cost.append(privacy)
        self.round_energy_cost.append(energy)
        self.round_model_cost.append(model_cost)
        self.round_total_cost.append(total)
        self.total_train_cost.append(total)
        self.decision_history.append(decisions)
        diagnostics = [client.optimizer.round_diagnostics for client in self.selected_clients
                       if getattr(client.optimizer, 'round_diagnostics', None) is not None]
        if diagnostics:
            self.mechanism_history.append(np.mean(diagnostics, axis=0).tolist())
        if self.convergence_round is None and accuracy >= self.target_acc:
            self.convergence_round = round_index
        print(f'ROUND {round_index + 1}/{self.global_rounds} accuracy={accuracy:.6f} test_loss={test_loss:.6f} cost={total:.6f}', flush=True)
        os.makedirs(os.path.join(results_root, 'metrics'), exist_ok=True)
        with open(os.path.join(results_root, 'metrics', f'{self.goal}.jsonl'), 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(round=round_index+1, accuracy=accuracy, test_loss=test_loss,
                total_cost=total, privacy_cost=privacy, energy_cost=energy, model_cost=model_cost)) + '\n')
        if len(self.rs_test_acc) % 10 == 0:
            self.save_results()
            os.makedirs(os.path.join(results_root, 'checkpoints'), exist_ok=True)
            path = os.path.join(results_root, 'checkpoints', f'{self.goal}_latest.pt')
            state = dict(round_completed=round_index+1,
                global_model={k: v.detach().cpu() for k, v in self.global_model.state_dict().items()},
                numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                clients=[dict(id=c.id, decision=getattr(c, 'decision_l', None),
                    scaler=getattr(c, 'scaler_c_i', None),
                    cc=[v.detach().cpu() for v in getattr(c, 'client_c', [])]) for c in self.clients],
                sc=[v.detach().cpu() for v in getattr(self, 'global_c', [])],
                source_sha256=os.environ.get('FL_SOURCE_SHA256', 'local'))
            if getattr(self, '_last_global_delta', None) is not None:
                state['last_global_delta'] = self._last_global_delta.detach().cpu()
            if hasattr(self, 'next_local_models'):
                state['next_local_models'] = self.next_local_models
            if torch.cuda.is_available():
                state['cuda_rng'] = torch.cuda.get_rng_state_all()
            torch.save(state, path + '.partial')
            os.replace(path + '.partial', path)

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        # print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.args, self.dataset, i, is_train=True)
            test_data = read_client_data(self.args, self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=False,
                               send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def save_instant_result(self, data, data_name):
        algo = self.args.model_name + "_" + self.dataset + "_" + self.algorithm + "_" + str(data_name) + "_"
        result_path = os.path.join(results_root, "client_selection", self.dataset, self.args.model_name) + os.sep

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file_path = result_path + "{}.txt".format(algo)

        with open(file_path, 'a', encoding='utf-8') as file:
            file.truncate(0)
            file.write(str(data) + "\n")

    def search_next_TRound(self, loss_array, target_loss):
        # a = len(loss_array) - 1
        # # a = len(loss_array[0:50])
        # x = np.arange(0, a, 1)
        # z1 = np.polyfit(x, loss_array[0:a], 6)  # 用3次多项式拟合  可以改为5 次多项式。。。。 返回三次多项式系数
        # # print(z1)
        # coefficients = []
        # for i in range(len(z1) - 1):
        #     coefficients.append(z1[i])
        # coefficients.append(z1[6] - target_loss)
        # # print(coefficients)
        # x_index = np.roots(coefficients)
        # x_index = np.real(x_index)
        # print(x_index)
        # min_ = 10000
        # count = 0
        # for x in x_index:
        #     count += 1
        #     if x > a and min_ > x:
        #         min_ = x
        # T = min_
        # if min_ >= 3000 and count == 3:
        #     T = 500

        x_data = np.array(range(0, len(loss_array)))  # x 的值
        y_data = np.array(loss_array)  # 对应的 y 值

        popt, pcov = curve_fit(self.sqrt_fit, x_data, y_data, p0=[1, 0])

        # popt 包含拟合参数 a 和 b
        a_fit, b_fit = popt

        # 使用拟合参数计算拟合曲线的 y 值
        T = ((target_loss - b_fit) / a_fit) ** 2

        return T, 0.01

    def sqrt_fit(self, x, a, b):
        return a * np.sqrt(x) + b

    def init_cs(self):
        clients_cs = []
        for i in range(0, 100):
            clients_cs.append([torch.zeros_like(param) for param in self.global_model.parameters()])

        self.center_cs = clients_cs
