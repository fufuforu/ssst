# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .input_types import (
    EncoderLatent,
    ModelInput,
    ModelInputDecoder,
    ModelInputEncoder,
    ModelSupervision,
    Reconstruction,
    split_data,
)
from .tokengs import TokenGS
from .siu3r_joint_ssst import SIU3RJointSSST
from .canonical_recon_models import LocusGSRecon, PlainTokenGSCanonicalRecon
from .object_locusgs import LocusGSObjectRecon
from .group_locusgs import LocusGSGroupRecon

# Model registry
model_registry = {
    'tokengs': TokenGS,
    'siu3r_joint_ssst': SIU3RJointSSST,
    'siu3r_locusgs_recon': LocusGSRecon,
    'siu3r_object_locusgs_recon': LocusGSObjectRecon,
    'siu3r_group_locusgs_recon': LocusGSGroupRecon,
    'siu3r_plain_tokengs_canonical_recon': PlainTokenGSCanonicalRecon,
}

# Export for convenience
__all__ = [
    'TokenGS',
    'split_data',
    'ModelInput',
    'ModelInputEncoder',
    'ModelInputDecoder',
    'ModelSupervision',
    'Reconstruction',
    'EncoderLatent',
    'SIU3RJointSSST',
    'LocusGSRecon',
    'LocusGSObjectRecon',
    'LocusGSGroupRecon',
    'PlainTokenGSCanonicalRecon',
    'model_registry',
]
