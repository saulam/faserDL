"""
Description: drop-in fallback for torch_ema.ExponentialMovingAverage (v0.3).

Used only when torch_ema is not installed (e.g. the neutrino-jepa
environment).  The update rule, warm-up (num_updates) and state_dict format
match torch_ema 0.3, so checkpoints are interchangeable.
"""

import copy
import weakref

import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params = None
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(self, parameters):
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError("EMA parameters were garbage collected; pass them explicitly.")
            return parameters
        parameters = list(parameters)
        if len(parameters) != len(self.shadow_params):
            raise ValueError("Number of parameters differs from the number of shadow parameters.")
        return parameters

    def update(self, parameters=None):
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, parameters):
                tmp = s_param - param
                tmp.mul_(one_minus_decay)
                s_param.sub_(tmp)

    def copy_to(self, parameters=None):
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None):
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict):
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = list(state_dict["shadow_params"])
        self.collected_params = state_dict["collected_params"]
        if len(self.shadow_params) == len(self._params_refs):
            params = [p() for p in self._params_refs]
            if not any(p is None for p in params):
                self.shadow_params = [
                    s.to(device=p.device, dtype=p.dtype) for s, p in zip(self.shadow_params, params)
                ]
