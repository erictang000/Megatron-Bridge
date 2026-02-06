# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM 4.7-Flash Model Provider.

GLM 4.7-Flash uses the DeepSeek-V3 architecture with Multi-Latent Attention (MLA).
"""

from dataclasses import dataclass, field
from typing import List, Union

from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class GLM47FlashModelProvider(MLAModelProvider):
    """
    GLM-4.7-Flash MoE Lite Model Provider.

    GLM 4.7-Flash uses the same MLA (Multi-Latent Attention) and MoE architecture
    as DeepSeek-V3, with different hyperparameters.

    HuggingFace: https://huggingface.co/zai-org/GLM-4.7-Flash
    """

    # Core architecture
    num_layers: int = 47  # N_DENSE_LAYERS (1) + N_MOE_LAYERS (46)
    hidden_size: int = 2048
    ffn_hidden_size: int = 10240
    num_attention_heads: int = 20
    vocab_size: int = 154880

    # Attention config
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    add_qkv_bias: bool = True  # QKV projections have bias
    share_embeddings_and_output_weights: bool = False
    qk_layernorm: bool = True
    multi_latent_attention: bool = True

    # MLA (Multi-Latent Attention) config
    kv_channels: int = 192
    q_lora_rank: int = 768
    kv_lora_rank: int = 512
    qk_head_dim: int = 192
    v_head_dim: int = 256
    qk_pos_emb_head_dim: int = 64
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    rotary_base: float = 1000000

    # MoE config
    num_moe_experts: int = 64
    moe_ffn_hidden_size: int = 1536
    moe_shared_expert_intermediate_size: int = 1536  # MOE_FFN_HIDDEN * MOE_SHARED_EXPERTS (1536 * 1)
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 46
    )  # first layer is dense, rest are MoE
    moe_router_topk: int = 4
    moe_router_topk_scaling_factor: float = 1.8
    moe_aux_loss_coeff: float = 0.0
    moe_router_score_function: str = "sigmoid"
    moe_router_pre_softmax: bool = True
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0.0
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_router_dtype: str = "fp32"
    moe_grouped_gemm: bool = True
    moe_permute_fusion: bool = True
    moe_shared_expert_overlap: bool = True


# Legacy alias for backward compatibility
GLM47FlashProvider = GLM47FlashModelProvider

