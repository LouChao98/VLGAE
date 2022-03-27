from __future__ import annotations

import torch

from .base import EncoderBase


class MultiEncoder(EncoderBase):
    """Compose encoders to different output."""

    def __init__(self, embedding, mapping, ff=None, **encoders):
        """
        :param mapping: a dict indicate show to construct x.
            for example: mapping = {
                'arc': ['shared_lstm.x', 'arc_lstm.x'],
                'rel':  ['shared_lstm.x', 'rel_lstm.x']
            }
        :param ff: a dict indicate passthrough variables
            e.g. ff = {
                'hiddens': 'shared_lstm.hiddens'
            }
        :type mapping: dict
        """
        super().__init__(embedding)

        self.all_encoders = []
        for key, value in encoders.items():
            if key.startswith('_'):
                continue
            self.add_module(key, value)
            self.all_encoders.append(key)

        self.mapping = {}  # {'shared_lstm': {'x': ['arc', 'rel']}, ...}
        self.output_fields = list(mapping.keys())
        self.dims = {o: 0 for o in self.output_fields}
        self.detailed_dims = {o: [] for o in self.output_fields}
        for target, sources in mapping.items():
            for source in sources:
                source_name, source_field = source.split('.')
                self.dims[target] += encoders[source_name].get_dim(source_field)
                self.detailed_dims[target].append(encoders[source_name].get_dim(source_field))
                if source_name not in self.mapping:
                    self.mapping[source_name] = {}
                if source_field not in self.mapping[source_name]:
                    self.mapping[source_name][source_field] = []
                self.mapping[source_name][source_field].append(target)
        self.ff = {}
        if ff is not None:
            for target, source in ff.items():
                source_name, source_field = source.split('.')
                assert target not in mapping, 'Conflict'
                if source_name not in self.ff:
                    self.ff[source_name] = {}
                if source_field not in self.ff[source_name]:
                    self.ff[source_name][source_field] = []
                self.ff[source_name][source_field].append(target)

    def forward(self, x, ctx):
        outputs = {key: [] for key in self.output_fields}
        for source_name in self.all_encoders:
            encoder_out = getattr(self, source_name)(x, ctx)
            if source_name in self.mapping:
                for encoder_field, targets in self.mapping[source_name].items():
                    for target in targets:
                        outputs[target].append(encoder_out[encoder_field])
            if source_name in self.ff:
                for encoder_field, targets in self.ff[source_name].items():
                    for target in targets:
                        outputs[target] = encoder_out[encoder_field]
        outputs = {
            key: torch.cat(value, dim=-1) if key in self.output_fields else value
            for key, value in outputs.items()
        }

        return outputs

    def get_dim(self, field):
        return self.dims[field]

