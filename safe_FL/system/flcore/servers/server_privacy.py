"""Minimal FedCEO and AdapLDP-FL adapters for comparable baseline runs."""

import torch

from flcore.clients.client_privacy import clientAdapLDP, clientFedCEO, clientLDP
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbase import Server


def add_privacy_baseline_args(parser):
    """Register only the knobs used by the two replacement baselines."""
    parser.add_argument("--fedceo-noise-multiplier", type=float, default=None)
    parser.add_argument("--fedceo-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fedceo-lamb", type=float, default=0.6)
    parser.add_argument("--fedceo-r", type=float, default=1.04)
    parser.add_argument("--fedceo-interval", type=int, default=10)
    parser.add_argument("--adapldp-noise-multiplier", type=float, default=None)
    parser.add_argument("--adapldp-clip", type=float, default=1.0)
    parser.add_argument(
        "--adapldp-mechanism", choices=("gaussian", "laplace"), default="gaussian"
    )
    parser.add_argument("--adapldp-direction-rate", type=float, default=0.5)
    parser.add_argument("--ldp-noise-multiplier", type=float, default=None)
    parser.add_argument("--ldp-clip", type=float, default=1.0)


def _weighted_mean(rows, weights):
    weight = torch.as_tensor(weights, device=rows.device, dtype=rows.dtype)
    weight = weight / weight.sum().clamp_min(1e-12)
    return weight @ rows


def _low_rank_smooth(rows, lamb, rank_growth, weights=None):
    """Shrink client-space singular values of stacked client models.

    The Gram matrix is only ``clients x clients``; this avoids a
    ``parameters x parameters`` matrix for ResNet-sized models.
    """
    if rows.shape[0] < 2:
        return _weighted_mean(rows, weights) if weights is not None else rows
    gram = rows @ rows.t()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    singular = eigenvalues.clamp_min(0).sqrt()
    threshold = max(float(lamb), 0.0) * max(float(rank_growth), 1.0)
    factors = torch.where(
        singular > 1e-12,
        (singular - threshold).clamp_min(0.0) / singular.clamp_min(1e-12),
        torch.zeros_like(singular),
    )
    transform = (eigenvectors * factors[None, :]) @ eigenvectors.t()
    if weights is not None:
        weight = torch.as_tensor(weights, device=rows.device, dtype=rows.dtype)
        weight = weight / weight.sum().clamp_min(1e-12)
        # Apply client-space smoothing to the weights before the large model matrix.
        return (weight @ transform) @ rows
    return transform @ rows


class FedCEO(FedAvg):
    """FedCEO-compatible low-rank semantic smoothing after local updates."""

    def __init__(self, args, times):
        self._fedceo_round = 0
        super().__init__(args, times)
        self.baseline_metadata = {
            "implementation": "FedCEO-compatible adapter",
            "aggregation": "client-space low-rank singular-value shrinkage",
            "upstream": "https://github.com/6lyc/FedCEO_Collaborate-with-Each-Other",
            "noise_multiplier": getattr(args, "fedceo_noise_multiplier", None)
            if getattr(args, "fedceo_noise_multiplier", None) is not None
            else getattr(args, "noise_multiplier", 0.0),
            "max_grad_norm": getattr(args, "fedceo_max_grad_norm", 1.0),
            "lamb": getattr(args, "fedceo_lamb", 0.6),
            "r": getattr(args, "fedceo_r", 1.04),
            "interval": getattr(args, "fedceo_interval", 10),
        }

    def set_clients(self, _client_obj=None):
        # FedAvg.__init__ calls set_clients(clientAVG); select the adapter
        # without duplicating the parent constructor or training loop.
        return Server.set_clients(self, clientFedCEO)

    def restore_training_state(self, checkpoint_path, curves_path):
        AdapLDPFL.restore_training_state(self, checkpoint_path, curves_path)
        self._fedceo_round = self.start_round

    def aggregate_parameters(self):
        assert self.uploaded_models
        self._fedceo_round += 1
        names = [name for name, _ in self.global_model.named_parameters()]
        client_parameters = [dict(model.named_parameters()) for model in self.uploaded_models]
        rows = torch.stack([
            torch.cat([parameters[name].detach().reshape(-1) for name in names])
            for parameters in client_parameters
        ])
        interval = max(int(getattr(self.args, "fedceo_interval", 10)), 1)
        smoothing = bool(getattr(self.args, "fedceo_enable_smoothing", True))
        if smoothing and self._fedceo_round % interval == 0 and rows.shape[0] > 1:
            aggregate = _low_rank_smooth(
                rows,
                getattr(self.args, "fedceo_lamb", 0.6),
                getattr(self.args, "fedceo_r", 1.04) ** (self._fedceo_round // interval),
                weights=self.uploaded_weights,
            )
        else:
            aggregate = _weighted_mean(rows, self.uploaded_weights)
        offset = 0
        with torch.no_grad():
            for name, parameter in self.global_model.named_parameters():
                size = parameter.numel()
                parameter.copy_(aggregate[offset : offset + size].view_as(parameter))
                offset += size


class AdapLDPFL(FedAvg):
    """AdapLDP-FL-compatible adaptive noise and direction correction."""

    def __init__(self, args, times):
        self._last_global_delta = None
        super().__init__(args, times)
        self.baseline_metadata = {
            "implementation": "AdapLDP-FL-compatible adapter",
            "aggregation": "adaptive update noise and global-direction correction",
            "upstream": "https://github.com/liyan2015/AdapLDP-FL",
            "noise_multiplier": getattr(args, "adapldp_noise_multiplier", None)
            if getattr(args, "adapldp_noise_multiplier", None) is not None
            else getattr(args, "noise_multiplier", 0.0),
            "clip": getattr(args, "adapldp_clip", 1.0),
            "mechanism": getattr(args, "adapldp_mechanism", "gaussian"),
            "direction_rate": getattr(args, "adapldp_direction_rate", 0.5),
        }

    def set_clients(self, _client_obj=None):
        return Server.set_clients(self, clientAdapLDP)

    def restore_training_state(self, checkpoint_path, curves_path):
        import h5py
        import numpy as np
        import random
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        completed = int(state['round_completed'])
        if not 0 < completed < self.global_rounds:
            raise ValueError('checkpoint must be a partial run')
        if self.args.learning_rate_decay or self.args.random_join_ratio or self.args.client_drop_rate:
            raise ValueError('resume only supports the fixed, stateless-SGD experiment protocol')
        with h5py.File(curves_path, 'r') as curves:
            expected = dict(algorithm=self.algorithm, dataset=self.dataset, model=self.args.model_name,
                            num_clients=self.num_clients, join_ratio=self.join_ratio, batch_size=self.batch_size,
                            local_epochs=self.local_epochs, local_learning_rate=self.learning_rate,
                            seed=self.args.seed, rounds_requested=self.global_rounds,
                            noise_multiplier=self.args.noise_multiplier)
            for name, value in expected.items():
                if curves.attrs[name] != value:
                    raise ValueError(f'resume protocol mismatch: {name}')
            if curves.attrs['source_sha256'] != state['source_sha256']:
                raise ValueError('checkpoint and curves source hashes differ')
            if len(curves['rs_test_acc']) != completed or len(state['clients']) != len(self.clients):
                raise ValueError('checkpoint and curves lengths differ')
            for name in ('rs_test_acc','rs_test_loss','rs_test_auc','rs_train_loss','eval_rounds',
                         'round_selected_count','decision_history','total_train_cost','round_total_cost',
                         'round_privacy_cost','round_energy_cost','round_model_cost'):
                if name in curves:
                    setattr(self, name, np.asarray(curves[name]).tolist())
            for client, saved in zip(self.clients, state['clients']):
                if client.id != saved['id']:
                    raise ValueError('client order differs')
                client.scaler_c_i = saved['scaler']
                if saved['decision'] is not None:
                    client.decision_l = saved['decision']
                if hasattr(client, 'client_c'):
                    client.client_c = [value.to(self.device) for value in saved['cc']]
                for field in ('global_acc','local_acc','train_cost_history','modelAcc_history',
                              'privacy_history','trained_cost','energy_cost_history'):
                    key = f'Client_{client.id}_{field}'
                    attribute = {'global_acc':'global_acc_history','local_acc':'local_acc_history',
                                 'train_cost_history':'train_cost','trained_cost':'trained_cost'}.get(field,field)
                    if key in curves:
                        setattr(client, attribute, np.asarray(curves[key]).tolist())
            convergence = int(curves['convergence_round'][()])
            self.convergence_round = None if convergence < 0 else convergence
            if self.algorithm in ('Adap-CTA','FedPVR'):
                names=('raw_update_norm','corrected_update_norm','cc_norm','sc_norm','sc_cc_norm','correction_magnitude')
                self.mechanism_history = np.column_stack([np.asarray(curves['round_'+name]) for name in names]).tolist()
        self.global_model.load_state_dict(state['global_model'])
        if 'last_global_delta' in state:
            self._last_global_delta = state['last_global_delta'].to(self.device)
        if hasattr(self, 'global_c'):
            self.global_c = [value.to(self.device) for value in state['sc']]
        if self.algorithm == 'FedMut':
            self.next_local_models = state['next_local_models']
        if self.algorithm == 'Adap-CTA':
            self.average_decision_l = max(self.min_decision,self.total_layer-completed)
        self.start_round = completed
        if hasattr(self,'baseline_metadata'):
            self.baseline_metadata['resume_from_round'] = completed
            self.baseline_metadata['resume_original_source_sha256'] = state['source_sha256']
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'RESUMED from round {completed}; {self.global_rounds-completed} rounds remain', flush=True)

    def aggregate_parameters(self):
        assert self.uploaded_models
        old_parameters = [p.detach().clone() for p in self.global_model.parameters()]
        reference = self._last_global_delta
        direction_rate = min(
            max(float(getattr(self.args, "adapldp_direction_rate", 0.5)), 0.0), 1.0
        )
        aggregate_delta = None
        weight_sum = sum(self.uploaded_weights)
        for model, weight in zip(self.uploaded_models, self.uploaded_weights):
            updates = [
                (new.detach() - old).reshape(-1)
                for new, old in zip(model.parameters(), old_parameters)
            ]
            row = torch.cat(updates)
            if reference is not None and direction_rate:
                ref = reference.to(row.device)
                ref_norm = torch.dot(ref, ref).clamp_min(1e-12)
                coefficient = torch.dot(row, ref) / ref_norm
                projection = coefficient * ref
                row = (1.0 - direction_rate) * row + direction_rate * projection
            if aggregate_delta is None:
                aggregate_delta = torch.zeros_like(row)
            aggregate_delta.add_(row, alpha=float(weight / weight_sum))
        offset = 0
        with torch.no_grad():
            for parameter, old in zip(self.global_model.parameters(), old_parameters):
                size = parameter.numel()
                parameter.copy_(old + aggregate_delta[offset : offset + size].view_as(parameter))
                offset += size
        self._last_global_delta = aggregate_delta.detach().clone()


class LDPFL(FedAvg):
    """Executable LDP-FL-compatible client-update baseline.

    The inherited source contains LDP-FL artifacts but no runnable class.  It
    therefore uses the existing FedAvg loop with a distinct client-side
    clipped/randomized update and records that compatibility status in H5.
    """

    def __init__(self, args, times):
        super().__init__(args, times)
        self.baseline_metadata = {
            "implementation": "LDP-FL-compatible client-update randomizer",
            "aggregation": "sample-weighted FedAvg after client clipping/randomization",
            "clip": getattr(args, "ldp_clip", 1.0),
            "noise_multiplier": getattr(args, "ldp_noise_multiplier", None)
            if getattr(args, "ldp_noise_multiplier", None) is not None
            else getattr(args, "noise_multiplier", 0.0),
            "source_status": "inherited source had logs/models but no executable LDP-FL class",
        }

    def set_clients(self, _client_obj=None):
        return Server.set_clients(self, clientLDP)


if __name__ == "__main__":
    # The FedCEO smoothing step must affect the aggregate even for equal-size clients.
    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
    smoothed = _low_rank_smooth(rows, 0.1, 1.0)
    assert not torch.allclose(
        _weighted_mean(rows, [1, 1, 1]),
        _weighted_mean(smoothed, [1, 1, 1]),
    )
