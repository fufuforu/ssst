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
from .instance_state_locusgs import LocusGSInstanceStateRecon
from .anchor_group_locusgs import LocusGSAnchorGroupRecon
from .object_locus_v1 import LocusGSObjectLocusV1Recon
from .object_locus_v2 import LocusGSObjectLocusV2Recon
from .object_locus_v2_1 import LocusGSObjectLocusV2_1Recon
from .object_locus_v3_set import LocusGSObjectLocusV3SetRecon

from .official_locusgs_recon import OfficialLocusGSRecon

# Model registry
model_registry = {
    'siu3r_official_locusgs_recon': OfficialLocusGSRecon,
    'tokengs': TokenGS,
    'siu3r_joint_ssst': SIU3RJointSSST,
    'siu3r_locusgs_recon': LocusGSRecon,
    'siu3r_object_locusgs_recon': LocusGSObjectRecon,
    'siu3r_group_locusgs_recon': LocusGSGroupRecon,
    'siu3r_instance_state_locusgs': LocusGSInstanceStateRecon,
    'siu3r_anchor_group_locusgs': LocusGSAnchorGroupRecon,
    'siu3r_object_locus_v1': LocusGSObjectLocusV1Recon,
    'siu3r_object_locus_v2': LocusGSObjectLocusV2Recon,
    'siu3r_object_locus_v2_1': LocusGSObjectLocusV2_1Recon,
    'siu3r_object_locus_v3_set': LocusGSObjectLocusV3SetRecon,
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
    'LocusGSInstanceStateRecon',
    'LocusGSAnchorGroupRecon',
    'LocusGSObjectLocusV1Recon',
    'LocusGSObjectLocusV2Recon',
    'LocusGSObjectLocusV2_1Recon',
    'LocusGSObjectLocusV3SetRecon',
    'PlainTokenGSCanonicalRecon',
    'model_registry',
]
