"""Run with Python: check masked SGD, the paper potential, and streaming aggregation."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'system'))
from flcore.optimizers.fedoptimizer import S_SCAFFOLDOptimizer
from flcore.servers.saveserver import exp
from flcore.servers.server_privacy import AdapLDPFL
from flcore.servers.serverbase import Server
from flcore.servers.serverS_scaffold import S_SCAFFOLD
import flcore.servers.serverbase as serverbase
from flcore.trainmodel.transformer import TransformerModel

parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
optimizer = S_SCAFFOLDOptimizer(parameters, lr=0.1, total_layer=3)
for parameter in parameters:
    parameter.grad = torch.full_like(parameter, 2)
optimizer.step([torch.full_like(p, 3) for p in parameters], [torch.ones_like(p) for p in parameters], 1)
assert torch.allclose(torch.cat(parameters), torch.tensor([0.8, 0.8, 0.6]))
assert np.isfinite(optimizer.round_diagnostics).all()

# A selected subset must preserve the mean of all client controls, including
# nonparticipating clients, when the initial controls are nonzero.
control_server=object.__new__(Server)
control_server.args=SimpleNamespace(model=torch.nn.Linear(1,1,bias=False),target_accuracy=.5)
control_server.global_model=control_server.args.model
control_server.num_clients=4
control_server.dataset='control_check'
control_server.algorithm='SCAFFOLD'
control_server.clients=[]
control_server.train_slow_clients=control_server.send_slow_clients=[False]*4
def control_client(args,id,**kwargs):
    model=copy.deepcopy(args.model)
    return SimpleNamespace(id=id,model=model,train_samples=1,test_samples=1,
                           client_c=[torch.full_like(p,float(id+1)) for p in model.parameters()])
read_client_data=serverbase.read_client_data
try:
    serverbase.read_client_data=lambda *a,**k: [0]
    control_server.set_clients(control_client)
finally:
    serverbase.read_client_data=read_client_data
assert torch.allclose(control_server.global_c[0],torch.tensor([[2.5]]))
for server_type in (exp,S_SCAFFOLD):
    partial=object.__new__(server_type)
    partial.global_model=torch.nn.Linear(1,1,bias=False)
    partial.global_model.weight.data.fill_(1)
    partial.global_c=[torch.tensor([[2.5]])]
    partial.num_clients,partial.num_join_clients,partial.server_learning_rate=4,2,1
    partial.selected_clients=[SimpleNamespace(delta_yc=lambda dy=dy,dc=dc:
        ([torch.tensor([[dy]])],[torch.tensor([[dc]])])) for dy,dc in ((1.,4.),(3.,-2.))]
    partial.aggregate_parameters()
    assert torch.allclose(partial.global_model.weight,torch.tensor([[3.]]))
    assert torch.allclose(partial.global_c[0],torch.tensor([[(5.+2.+3.+2.)/4]]))

game = object.__new__(exp)
game.total_layer, game.min_decision = 27, 6
game.global_model = torch.nn.Linear(1, 1, bias=False)
game.args = SimpleNamespace(num_clients=100, batch_size=64)
game.rs_test_acc = [0.0]
game.selected_clients = [SimpleNamespace(decision_l=value, train_samples=64,
                                          num_batches=1, target_acc=.5)
                         for value in (6, 12, 27)]
decisions = [c.decision_l for c in game.selected_clients]
game.make_clients_decision(decisions)
assert all(game.min_decision <= value <= game.total_layer for value in decisions)
again=decisions.copy(); game.make_clients_decision(again)
assert all(game.min_decision <= value <= game.total_layer for value in again)

server = object.__new__(AdapLDPFL)
server.args = SimpleNamespace(adapldp_direction_rate=.5)
server.global_model = torch.nn.Linear(2, 2, bias=False)
server.global_model.weight.data.fill_(1)
server._last_global_delta = torch.tensor([0., 1., 1., 0.])
server.uploaded_weights = [.25, .75]
server.uploaded_models = [copy.deepcopy(server.global_model) for _ in range(2)]
for model, value in zip(server.uploaded_models, (3, 5)):
    model.weight.data.fill_(value)
server.aggregate_parameters()
assert torch.allclose(server.global_model.weight, torch.tensor([[2.75, 4.5], [4.5, 2.75]]))

model=TransformerModel(80,64,4,2,80,d_hid=128,pooling='last')
output=model((torch.randint(80,(2,80)),torch.tensor([80,65])))
loss=torch.nn.functional.cross_entropy(output,torch.tensor([0,79]))
loss.backward()
assert output.shape==(2,80) and all(torch.isfinite(p.grad).all() for p in model.parameters())

print('PASS: masked correction, consistent control initialization and subset updates, finite diagnostics, stable best responses, equivalent streaming aggregation, text backward')
