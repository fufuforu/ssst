# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import stand-in for the fused-ssim CUDA extension.

`tokengs.rendering` imports this symbol on CUDA machines, but the extension is
built from a GitHub source that this cluster cannot reach. SSST trains with
`lambda_ssim = 0` and never evaluates the SSIM loss, so the stand-in only needs
to satisfy the import. Any actual use fails loudly instead of degrading.
"""


def fusedssim(*_args, **_kwargs):
    raise NotImplementedError(
        "fused-ssim is not built in this environment; SSST does not use the "
        "SSIM loss (lambda_ssim = 0). Install fused-ssim to enable it."
    )


def fusedssim_backward(*_args, **_kwargs):
    raise NotImplementedError(
        "fused-ssim is not built in this environment; SSST does not use the "
        "SSIM loss (lambda_ssim = 0). Install fused-ssim to enable it."
    )
