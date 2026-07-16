---
title: "第 15 章：NN Modules 完整 API 目录"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "覆盖 zeta/nn/modules/ 下全部 216 个模块文件。每个模块列出公开类及其 forward 等方法和公开函数。"
summary: "覆盖 zeta/nn/modules/ 下全部 216 个模块文件。每个模块列出公开类及其 forward 等方法和公开函数。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 16
series_weight: 16
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 15 章：NN Modules 完整 API 目录

> 覆盖 `zeta/nn/modules/` 下全部 **216** 个模块文件。每个模块列出公开类及其 `forward` 等方法和公开函数。  
> 算法原理见各专题章节；本章为 **API 速查，确保无遗漏**。

## 目录说明

| 分类 | 对应章节 |
|------|----------|
| 注意力/SSM | [05-ssm-mamba.md](./05-ssm-mamba.md)、[03-attention.md](./03-attention.md) |
| MoE | [06-moe.md](./06-moe.md) |
| 视觉/卷积 | [07-vision-conv.md](./07-vision-conv.md) |
| 多模态 | [08-multimodal.md](./08-multimodal.md) |
| 流匹配/扩散 | 下文 §Flow & Diffusion |
| 未导出模块 | [17-internal-modules.md](./17-internal-modules.md) |

## Flow & Diffusion 模块摘要

| 模块 | 符号 | 原理要点 |
|------|------|----------|
| `flow_matching.py` | `Flow`, `MixtureFlow`, `FlowConfig`, `MixtureFlowConfig` | 连续归一化流 / Flow Matching：学习 $v_\theta(x,t)$ 使 ODE 将噪声映射到数据 |
| `flow_transformer.py` | `FlowTransformer`, `FlowMLP`, `FlowTransformerConfig` | Transformer 骨干的 Flow 模型 |
| `diffusion.py` | `Diffuser` | DDPM 风格前向加噪 $q(x_t\|x_0)$ 与采样 |
| `vit_denoiser.py` | `VitTransformerBlock` | ViT 去噪骨干 |
| `video_diffusion_modules.py` | `TemporalDownsample`, `TemporalUpsample`, `ConvolutionInflationBlock`, `AttentionBasedInflationBlock` | 2D→3D 膨胀，视频扩散 |

**Flow Matching 损失**（概念）：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\| v_\theta(x_t, t) - (x_1 - x_0) \|^2\right]$$

其中 $x_t = (1-t)x_0 + t x_1$。**参考**：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

---

## 模块 API 列表（按文件名排序）

### _activations

- **PytorchGELUTanh**: A fast C implementation of the tanh approximation of the GeLU activation function. See https://arxiv.org/abs/1606.08415.
  - methods: forward
- **NewGELUActivation**: Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see the Gau
  - methods: forward
- **GELUActivation**: Original Implementation of the GELU activation function in Google BERT repo when initially created. For information: Ope
  - methods: forward
- **FastGELUActivation**: Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
  - methods: forward
- **QuickGELUActivation**: Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
  - methods: forward
- **ClippedGELUActivation**: Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as it al
  - methods: forward
- **AccurateGELUActivation**: Applies GELU approximation that is faster than default and more accurate than QuickGELU. See: https://github.com/hendryc
  - methods: forward
- **MishActivation**: See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also visit th
  - methods: forward
- **LinearActivation**: Applies the linear activation function, i.e. forwarding input directly to output.
  - methods: forward
- **LaplaceActivation**: Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See https://arx
  - methods: forward
- **ReLUSquaredActivation**: Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
  - methods: forward
- **ClassInstantier**
- **get_activation()**

### adaptive_conv

- **Residual**
  - methods: forward
- **AdaptiveConv3DMod**: Adaptive convolutional layer, with support for spatial modulation  Args:     dim: input channels     spatial_kernel: spa
  - methods: forward
- **exists()**
- **default()**
- **identity()**
- **divisible_by()**
- **pack_one()**
- **unpack_one()**
- **is_odd()**
- **cast_tuple()**
- **Sequential()**

### adaptive_gating

- **AdaptiveGating**
  - methods: forward
- **exists()**

### adaptive_layernorm

- **AdaptiveLayerNorm**: Adaptive Layer Normalization module.   Args:     num_features (int): number of features in the input tensor     eps (flo
  - methods: forward

### adaptive_parameter_list

- **AdaptiveParameterList**: A container that allows for parameters to adapt their values based on the learning process  Example:     ```     def ada
  - methods: adapt

### adaptive_rmsnorm

- **AdaptiveRMSNorm**: Adaptive Root Mean Square Normalization (RMSNorm) module.  Args:     dim (int): The input dimension.     dim_cond (int):
  - methods: forward
- **exists()**
- **append_dims()**

### add_norm

- **add_norm()**: _summary_  Args:     x (_type_): _description_     dim (int): _description_     

### alr_block

- **FeedForward**
  - methods: forward
- **ALRBlock**: ALRBlock class A transformer like layer that uses feedforward networks instead of self-attention  Args:     dim (int): I
  - methods: forward

### attn

- **scaled_dot_product_attention()**: Compute scaled dot product attention.  Args:     query (torch.Tensor): The query

### audio_to_text

- **audio_to_text()**: Reshapes and projects the audio input tensor to text representation.  Args:     

### avg_model_merger

- **AverageModelMerger**: A class to merge multiple models by averaging their weights.  This is a simple yet effective method to combine models tr
  - methods: merge_models

### batched_dp

- **batched_dot_product()**

### block_butterfly_mlp

- **BlockButterflyLinear**: BlockButterflyMLP is a module that applies a block butterfly transformation to the input tensor.  Args:     num_blocks (
  - methods: forward
- **BlockMLP**
  - methods: forward

### blockdiag_butterfly

- **BlockdiagButterflyMultiply**: This is a faster implementation, with careful memory copies for the fastest bmm performance. The backward pass is also w
  - methods: forward, backward
- **BlockdiagMultiply**: This is a faster implementation, with careful memory copies for the fastest bmm performance. The backward pass is also w
  - methods: forward, backward
- **Sin**
  - methods: forward
- **StructuredLinear**
  - methods: reset_parameters, set_weights_from_dense_init, reset_parameters_bias, saving, convert_to_dense_weight, preprocess, postprocess, forward_matmul, forward
- **blockdiag_butterfly_multiply_reference()**: This implementation is slow but more likely to be correct. There are 3 implement
- **blockdiag_weight_to_dense_weight()**: Argumments:     weight: (nblocks, out / nblocks, in / blocks) Return:     dense_
- **blockdiag_multiply_reference()**: This implementation is slow but more likely to be correct. Arguments:     x: (..
- **fftconv_ref()**
- **mul_sum()**

### chan_layer_norm

- **ChanLayerNorm**
  - methods: forward

### clex

- **ODELinear**
  - methods: reset_parameters, get_time_embedding, forward
- **Clex**: CLEx: Continuous Rotation Positional Encoding  Args:     dim: dimension of the input     max_position_embeddings: maximu
  - methods: sample_random_times, get_random_position_ids, get_continuous_freq, forward

### clip_bottleneck

- **ClipBottleneck**: ClipBottleneck is a bottleneck block with a stride of 1 and an avgpool layer after the second conv layer.  Args:     inp
  - methods: forward

### cnn_text

- **CNNNew**: CNN for language  Args: vocab_size: size of vocabulary embedding_dim: dimension of embedding n_filters: number of filter
  - methods: forward

### cog_vlm_two_adapter

- **CogVLMTwoAdapter**: CogVLMTwoAdapter module that reduces the sequence length of ViT outputs and aligns the features with linguistic represen
  - methods: forward

### combined_linear

- **CombinedLinear**: Applies a linear transformation to the incoming data: :math:`y = xA^T + b`  Compared to torch.nn.Linear, uses a combined
  - methods: reset_parameters, forward, extra_repr

### conv_bn_relu

- **ConvBNReLU**: A conv layer followed by batch normalization and ReLU activation.  Args:     in_planes (int): Number of input channels. 
  - methods: forward

### conv_mlp

- **Conv2DFeedforward**: A Convolutional feed-forward network, as proposed in VAN_ (Vision Attention Network, Guo et al.)  .. _VAN: https://arxiv
  - methods: init_weights, forward

### convnet

- **ConvNet**: Convolutional Neural Network for MNIST classification.  Usage:     net = ConvNet()     net(x)
  - methods: forward

### cope

- **CoPE**
  - methods: forward

### crome_adapter

- **CROMEAdapter**
  - methods: forward

### cross_embed_layer

- **CrossEmbedLayer**
  - methods: forward

### cross_modal_reparametization

- **CrossModalReparamLinear**: Linear layer with cross-modal reparameterization.  Args:     in_features (int): Size of each input sample.     out_featu
  - methods: forward
- **CrossModalReParametrization**: A module for cross-modal reparametrization.  Args:     original_linear (nn.Linear): The original linear layer.     auxil
  - methods: forward, merge
- **cross_modal_ffn()**: Cross-modal feed-forward network.  Args:     ffn_original_linear (nn.Linear): Li
- **build_cross_modal_reparam_linear()**
- **change_original_linear_to_reparam()**
- **reparameterize_aux_into_target_model()**: Reparameterizes the auxiliary model into the target model by replacing specific 

### decision_tree

- **SimpleDecisionTree**: Simple decision tree model with residual connections and multi head output.   Args:     input_size (int): Input size of 
  - methods: forward

### deepseek_moe

- **DeepSeekMoE**
  - methods: forward

### dense_connect

- **DenseBlock**
  - methods: forward

### diffusion

- **Diffuser**: Implements the diffusion process for image tensors, progressively adding Gaussian noise.  Attributes:     num_timesteps 
  - methods: forward

### droppath

- *(no public symbols)*

### dual_path_block

- **DualPathBlock**
  - methods: forward

### dyna_conv

- **DynaConv**: DynaConv dynamically generates convolutional kernels based on the input features.  This layer replaces traditional convo
  - methods: reset_parameters, forward

### dynamic_module

- **DynamicModule**: A container that allows for dynamic addition, removal, and modification of modules  examples ```` dynamic_module = Dynam
  - methods: add, remove, forward, save_state, load_state

### dynamic_routing_block

- **DynamicRoutingBlock**
  - methods: forward

### embedding_to_grid

- **embedding_to_grid()**: Embedding to grid  Einstein summation notation:     'b' = batch size     'h' = i

### ether

- **Ether**: **Algorithmic Pseudocode for MMOLF**:  1. **Inputs**:     - \( y_{pred} \) (Predicted values from the model)     - \( y_
  - methods: forward

### evlm_xattn

- **GatedXAttention**: GatedXAttention module applies cross attention between text and image embeddings, followed by activation functions and f
  - methods: forward
- **GatedMoECrossAttn**: GatedMoECrossAttn is a module that performs gated multi-expert cross attention on text and image inputs.  Args:     dim 
  - methods: forward

### exo

- **Exo**: Exo activation function -----------------------  Exo is a new activation function that is a combination of linear and no
  - methods: forward

### expand

- **expand()**

### expand_channels

- **expand_channels()**: Expand the channel dimenions of a tensor  Einstein summation notation:     'b' =

### expert

- **Experts**: Expert module for the Mixture of Experts layer.  Args:     dim (int): Dimension of the input features.     experts (int)
  - methods: forward

### fast_text

- **FastTextNew()**: FastText for language  Args: vocab_size: size of vocabulary embedding_dim: dimen

### feedback_block

- **FeedbackBlock**
  - methods: forward

### feedforward

- **ReluSquared**
  - methods: forward
- **FeedForward**
  - methods: forward
- **exists()**
- **default()**
- **init_zero_()**

### feedforward_network

- **set_torch_seed**
  - methods: get_rng_state, set_rng_state
- **FeedForwardNetwork**
  - methods: reset_parameters, forward
- **make_experts()**
- **get_activation_fn()**

### film

- **Film**: Feature-wise Linear Modulation (FiLM) module.  This module applies feature-wise linear modulation to the input features 
  - methods: forward

### film_conditioning

- **FilmConditioning**: FilmConditioning module applies feature-wise affine transformations to the input tensor based on conditioning tensor.  A
  - methods: forward

### film_efficient_metb3

- **FiLMEfficientNetB3**
  - methods: forward

### flash_conv

- **FlashFFTConvWrapper**

### flatten_features

- **flatten_features()**: Flaten the spatial dimensions of a feature map  Einstein summation notation:    

### flex_conv

- **FlexiConv**: FlexiConv is an experimental and flexible convolutional layer that adapts to the input data.  This layer uses parameteri
  - methods: forward

### flexible_mlp

- **CustomMLP**: A customizable Multi-Layer Perceptron (MLP).  Attributes:     layers (nn.ModuleList): List of linear layers.     activat
  - methods: forward

### flow_matching

- **FlowConfig**: Configuration for Flow neural network.  Attributes:     dim: Input/output dimension     hidden_dim: Hidden layer dimensi
- **Flow**: Neural network for modeling continuous normalizing flows.  This class implements a neural network that learns the veloci
  - methods: forward, step, train_model, visualize_flow, save_checkpoint, load_checkpoint
- **MixtureFlowConfig**: Configuration for Mixture of Flows neural network.  Attributes:     n_experts: Number of flow experts in the mixture    
- **GatingNetwork**: Neural network for selecting which flow expert to use.  The gating network takes the current state and time as input and
  - methods: forward
- **ExpertFlow**: Individual flow expert network.
  - methods: forward
- **MixtureFlow**: Mixture of Flows model combining multiple flow experts with a gating network.
  - methods: forward, step, train_model, save_checkpoint, load_checkpoint, visualize_flow, visualize_expert_specialization, get_expert_statistics
- **make_moons()**: Generate a 2D dataset with two interleaving half circles (moons).  This is a cus

### flow_transformer

- **FlowTransformerConfig**: Configuration for Flow Transformer.  Attributes:     dim: Model dimension     heads: Number of attention heads     depth
- **FlowMLP**
  - methods: forward
- **MultiHeadAttention**
  - methods: forward
- **TransformerBlock**: Transformer block with FlowMLP.
  - methods: forward
- **FlowTransformer**: Transformer model with FlowMLP blocks.
  - methods: forward, train_step, save_checkpoint, load_checkpoint
- **create_optimizer()**: Create optimizer with weight decay fix.  Args:     model: FlowTransformer model 

### fractoral_norm

- **FractoralNorm**: FractoralNorm module applies LayerNorm to the input tensor multiple times in a row.  Args:     dim (int): Number of feat
  - methods: forward

### fractorial_net

- **FractalBlock**
  - methods: forward
- **FractalNetwork**
  - methods: forward

### freeze_layers

- **set_module_requires_grad()**: Set the `requires_grad` attribute of all parameters in the given module.  Args: 
- **freeze_all_layers()**: Freezes all layers in the given module by setting their requires_grad attribute 

### fused_dropout_add

- **jit_dropout_add()**
- **fused_dropout_add()**: Applies fused dropout and addition operation to the input tensors.  Args:     x 
- **jit_bias_dropout_add()**: Applies dropout to the sum of input `x` and `bias`, and then adds the `residual`
- **fused_bias_dropout_add()**: Applies fused bias, dropout, and addition operation to the input tensor.  Args: 

### fused_dropout_layernom

- **FusedDropoutLayerNorm**: FusedDropoutLayerNorm  Args:     dim (int): Input dimension     dropout (float, optional): Dropout. Defaults to 0.1.    
  - methods: forward

### fused_gelu_dense

- **FusedDenseGELUDense**: FuseFusedDenseGELUDense  Args     dim (int): Input dimension     dim_out (int): Output dimension     bias (bool, optiona
  - methods: forward

### fusion_ffn

- **MMFusionFFN**: Positionwise feed forward layer.  Args:     dim (int): input dimension.     hidden_dim (int): hidden dimension.     drop
  - methods: forward

### g_shard_moe

- **MOELayer**: Base Mixture of Experts (MoE) layer implementation using pure PyTorch.  This class serves as a base class for MoE implem
  - methods: forward, prepare_for_inference_, get_aux_loss, get_metadata, reset_metadata
- **FastDispatcher**: Custom implementation for efficient token dispatching in MoE layers.  This class provides efficient dispatching of token
  - methods: update, encode, decode
- **Top1Gate**: Top-1 gating mechanism for MoE.
  - methods: forward
- **Top2Gate**: Top-2 gating mechanism for MoE.
  - methods: forward
- **_AllToAll**: All-to-all communication primitive.
  - methods: forward, backward
- **GShardMoELayer**: GShard Mixture of Experts (MoE) layer implementation using pure PyTorch.  This implementation follows the GShard paper a
  - methods: forward, prepare_for_inference_, all_to_all_wrapper, record_all_to_all_stats
- **fast_cumsum_sub_one()**: Compute cumulative sum along dimension 0 minus 1.
- **get_moe_group()**: Get the MoE process group for expert parallelism.
- **get_all2all_group()**: Get the all-to-all process group for MoE communication.
- **one_hot()**: Create one-hot encoding of indices.
- **entropy()**: Compute entropy of probability distributions.
- **gumbel_rsample()**: Sample from Gumbel distribution.
- **top1gating()**: Implements Top1 gating for MoE.
- **top2gating()**: Implements Top2 gating for MoE.

### gated_cnn_block

- **GatedCNNBlock**
  - methods: forward

### gated_residual_block

- **GatedResidualBlock**
  - methods: forward

### gill_mapper

- **GILLMapper**: GILLMapper is a module that maps image and text embeddings using a Transformer model. From the paper: "https://arxiv.org
  - methods: forward

### glu

- **GLU**: GLU (Gated Linear Unit) module.  Args:     dim_in (int): Input dimension.     dim_out (int): Output dimension.     activ
  - methods: forward

### gru_gating

- **Residual**
  - methods: forward
- **GRUGating**: GRUGating Overview: GRU gating mechanism  Args:     dim (int): dimension of the embedding     scale_residual (bool): whe
  - methods: forward
- **exists()**

### h3

- **DiagonalSSM**: DiagonalSSM is a module that implements the Diagonal SSM operation.  Args:     nn (_type_): _description_
  - methods: forward
- **ShiftSSM**: ShiftSSM is a module that implements the Shift SSM operation.  Args:     nn (_type_): _description_
  - methods: forward
- **H3Layer**: H3Layer is a layer that implements the H3 associative memory model.   Attributes:     dim (int): The dimensionality of t
  - methods: forward

### hebbian

- **BasicHebbianGRUModel**: A basic Hebbian learning model combined with a GRU for text-based tasks.  This model applies a simple Hebbian update rul
  - methods: forward

### highway_layer

- **HighwayLayer**
  - methods: forward

### image_projector

- **ImagePatchCreatorProjector**: Image Patch Creator and Projector Layer.  This layer dynamically creates and projects image patches suitable for feeding
  - methods: forward, calculate_dynamic_patch_size, create_patches

### image_to_text

- **img_to_text()**: Convert an image tensor to a text tensor.  Args:     x (Tensor): Input image ten

### img_or_video_to_time

- **exists()**
- **pack_one()**
- **unpack_one()**
- **compact_values()**
- **image_or_video_to_time()**: Decorator function that converts the input tensor from image or video format to 

### img_patch_embed

- **ImgPatchEmbed**: patch embedding module   Args:     img_size (int, optional): image size. Defaults to 224.     patch_size (int, optional)
  - methods: forward

### img_reshape

- **image_reshape()**: Reshapes an image batch  Einstein summation notation:     'b' = batch size     '

### itca

- **PreNorm**: Prenorm  Args:     dim (_type_): _description_     fn (_type_): _description_
  - methods: forward
- **CrossAttention**
  - methods: forward
- **IterativeCrossSelfAttention**: Iterative  Args:     dim (_type_): _description_     depth (_type_): _description_     heads (_type_): _description_    
  - methods: forward

### kv_cache

- **KVCache**: KVCache is a module that stores the key and value tensors for each position in the input sequence. This is used in the d
  - methods: update
- **find_multiple()**: Finds the smallest multiple of k that is greater than or equal to n.  Args:     
- **precompute_freq_cis()**: Precomputes the frequency values for the positional encodings.  Args:     seq_le
- **setup_cache()**: Sets up the cache for the given model.  Args:     max_batch_size (_type_): _desc

### kv_cache_update

- **kv_cache_with_update()**: Single-head KV cache update with Dynamic Memory Compression (DMC).  Parameters: 

### lambda_mask

- **LayerNorm**
  - methods: forward
- **FeedForward**
  - methods: forward
- **Attention**
  - methods: forward
- **Transformer**
  - methods: forward
- **MegaVit**: MegaVit model from https://arxiv.org/abs/2106.14759  Args: ------- image_size: int     Size of image patch_size: int    
  - methods: forward
- **pair()**

### lang_conv_module

- **ConvolutionLanguageBlock**: Convolutional block for language modeling. -------------------------------------------- A convolutional block that consi
  - methods: forward

### laser

- **Laser**: Layer Selective Rank Reduction (LASER) is a module that replaces specific weight matrices in a Transformer model by thei
  - methods: forward, low_rank_approximation

### layer_scale

- **LayerScale**: Applies layer scaling to the output of a given module.  Args:     fn (Module): The module to apply layer scaling to.    
  - methods: forward

### layernorm

- **LayerNorm**: Layer normalization module.  Args:     dim (int): The dimension of the input tensor.     eps (float, optional): A small 
  - methods: forward
- **l2norm()**: L2 normalization function.  Args:     t (torch.Tensor): The input tensor.  Retur

### leaky_relu

- **LeakyRELU**: LeakyReLU activation function.  Args:     nn (_type_): _description_  Returns:     _type_: _description_
  - methods: forward, extra_repr

### log_ff

- **LogFF**: An implementation of fast feedforward networks from the paper "Fast Feedforward Networks".  Args:     input_width (int):
  - methods: get_node_param_group, get_leaf_param_group, training_forward, forward, eval_forward
- **compute_entropy_safe()**: Computes the entropy of a Bernoulli distribution with probability `p`.  Paramete

### lora

- **Lora**: Lora module applies a linear transformation to the input tensor using the Lora algorithm.  Args:     dim (int): The inpu
  - methods: weight, forward

### matrix

- **Matrix**: Matrix class that can be converted between frameworks   Args:     data (torch.Tensor, jnp.ndarray, tf.Tensor): Data to b
  - methods: to_pytorch, to_jax, to_tensorflow, sum

### mbconv

- **DropSample**
  - methods: forward
- **SqueezeExcitation**: Squeeze-and-Excitation module for channel-wise feature recalibration.  Args:     dim (int): Number of input channels.   
  - methods: forward
- **MBConvResidual**
  - methods: forward
- **MBConv()**: MobileNetV3 Bottleneck Convolution (MBConv) block.  Args:     dim_in (int): Numb

### mixtape

- **Mixtape**
  - methods: forward

### mixtral_expert

- **MixtralExpert**: At every layer, for every token, a router network chooses two of these groups (the “experts”) to process the token and c
  - methods: forward

### mlp

- **MLP**: Multi-Layer Perceptron (MLP) module.  Args:     dim_in (int): The dimension of the input tensor.     dim_out (int): The 
  - methods: forward

### mlp_mixer

- **MLPBlock**: MLPBlock  Args:     dim (int): [description]
  - methods: forward
- **MixerBlock**: MixerBlock   Args:     mlp_dim (int): [description]     channels_dim (int): [description]
  - methods: forward
- **MLPMixer**: MLPMixer  Args:     num_classes (int): [description]     num_blocks (int): [description]     patch_size (int): [descript
  - methods: forward

### mm_adapter

- **SkipConnection**: A helper class for implementing skip connections.
  - methods: forward
- **MultiModalAdapterDenseNetwork**: Multi-modal adapter dense network that takes a tensor of shape (batch_size, dim) and returns a tensor of shape (batch_si
  - methods: forward

### mm_fusion

- **multi_modal_fusion()**

### mm_layernorm

- **MMLayerNorm**
  - methods: forward

### mm_ops

- **threed_to_text()**: Converts a 3D tensor to text representation.  Args:     x (Tensor): The input te
- **text_to_twod()**: Converts a 3D tensor of shape (batch_size, sequence_length, dim) to a 2D tensor 

### modality_adaptive_module

- **ModalityAdaptiveModule**: Modality Adaptive Module  Args:     dim: int         The dimension of the input features     heads: int         The numb
  - methods: modality_indicator, forward

### moe

- **MixtureOfExperts**: Mixture of Experts model.  Args:     dim (int): Input dimension.     num_experts (int): Number of experts in the mixture
  - methods: forward

### moe_router

- **MoERouter**: MoERouter is a module that routes input data to multiple experts based on a specified mechanism.  Args:     dim (int): T
  - methods: forward

### monarch_mlp

- **MonarchMLP**: A sparse MLP from this paper: https://hazyresearch.stanford.edu/blog/2024-01-11-m2-bert-retrieval  Example:     >>> x = 
  - methods: forward

### mr_adapter

- **MRAdapter**: Multi-Resolution Adapter module for neural networks.  Args:     dim (int): The input dimension.     heads (int, optional
  - methods: forward

### multi_input_multi_output

- **MultiModalEmbedding**: MultiModalEmbedding class represents a module for multi-modal embedding.  Args:     video_dim (int): The dimension of th
  - methods: forward
- **MultiInputMultiModalConcatenation**: A module that concatenates multiple input tensors along a specified dimension.  Args:     dim (int): The dimension along
  - methods: forward
- **SplitMultiOutput**: Splits the input tensor into multiple outputs along a specified dimension.  Args:     dim (int): The dimension along whi
  - methods: forward
- **OutputHead**
  - methods: forward
- **DynamicOutputDecoder**: Decoder module for dynamic output.  Args:     dim (int): The input dimension.     robot_count (int): The number of robot
  - methods: forward
- **DynamicInputChannels**: A module that applies linear transformations to input data for multiple robots.  Args:     num_robots (int): The number 
  - methods: forward
- **OutputDecoders**: Class representing the output decoders for multiple robots.  Args:     num_robots (int): The number of robots.     dim (
  - methods: forward

### multi_layer_key_cache

- **MultiLayerKeyValueAttention**: Multi-layer key-value attention module.  Args:     embed_size (int): The size of the input embeddings.     num_heads (in
  - methods: forward

### multi_scale_block

- **MultiScaleBlock**: A module that applies a given submodule to the input tensor at multiple scales.  Args:     module (nn.Module): The submo
  - methods: forward

### multiclass_label

- *(no public symbols)*

### multimodal_concat

- **multimodal_concat()**

### nearest_upsample

- **nearest_upsample()**: Nearest upsampling layer.  Args:     dim (int): _description_     dim_out (int, 

### nebula

- **HashableTensorWrapper**
- **LossFunction**
  - methods: compute_Loss
- **L1Loss**
  - methods: compute_loss
- **MSELoss**
  - methods: compute_loss
- **SmoothL1Loss**
  - methods: compute_Loss
- **MultiLabelSoftMarginLoss**
  - methods: compute_loss
- **PoissonNLLoss**
  - methods: compute_loss
- **KLDivLoss**
  - methods: compute_loss
- **NLLLoss**
  - methods: compute_loss
- **CrossEntropyLoss**
  - methods: compute_loss
- **Nebula**
  - methods: determine_loss_function
- **one_hot_encoding()**
- **is_multi_label_classification()**
- **contains_non_negative_integers()**
- **are_probability_distributions()**
- **are_log_probabilities()**
- **generate_tensor_key()**

### nfn_stem

- **NFNStem**: NFNStem module represents the stem of the NFN (Neural Filter Network) architecture.  Args:     in_channels (List[int]): 
  - methods: forward

### norm_fractorals

- **NormalizationFractral**: A module that performs normalization using fractal layers.  Args:     dim (int): The input dimension.     eps (float, op
  - methods: forward

### norm_utils

- **PreNorm**: Pre-normalization module that applies RMSNorm to the input before passing it through the given function.  Args:     dim 
  - methods: forward
- **PostNorm**: Post-normalization module that applies layer normalization after the input is passed through a given module.  Args:     
  - methods: forward

### omnimodal_fusion

- **OmniModalFusion**: OmniModalFusion is designed to fuse an arbitrary number of modalities with unknown shapes.  Args:     fusion_dim (int): 
  - methods: forward

### p_scan

- **PScan**: An implementation of the parallel scan operation in PyTorch (Blelloch version). This code is based on Francois Fleuret’s
  - methods: pscan, forward, backward

### palo_ldp

- **PaloLDP**: Implementation of the PaloLDP module.  Args:     dim (int): The dimension of the input tensor.     channels (int, option
  - methods: forward

### parallel_wrapper

- **Parallel**: A module that applies a list of functions in parallel and sums their outputs.  Args:     *fns: Variable number of functi
  - methods: forward

### patch_embedding_layer

- **PatchEmbeddingLayer**
  - methods: forward

### patch_img

- **patch_img()**

### patch_linear_flatten

- **posemb_sincos_2d()**
- **vit_output_head()**: Applies a Vision Transformer (ViT) output head to the input tensor.  Args:     x
- **patch_linear_flatten()**: Applies patch embedding to the input tensor and flattens it.  Args:     x (Tenso
- **video_patch_linear_flatten()**: Applies patch embedding to the input tensor and flattens it.  Args:     x (Tenso
- **cls_tokens()**: Adds class tokens to the input tensor and applies dropout and positional embeddi

### patch_video

- **patch_video()**: Patch a video into patches of size patch_size x patch_size x patch_size x C x H 

### peg

- **PEG**: PEG (Positional Encoding Generator) module.  Args:     dim (int): The input dimension.     kernel_size (int, optional): 
  - methods: forward

### perceiver_layer

- **PerceiverLayer**: Perceiver Layer, this layer has a self attn that takes in q then -> sends the output into the q of the cross attention w
  - methods: forward

### perceiver_resampler

- **PerceiverAttention**
  - methods: forward
- **PerceiverResampler**
  - methods: forward
- **MaskedCrossAttention**
  - methods: forward
- **GatedCrossAttentionBlock**
  - methods: forward
- **exists()**
- **FeedForward()**

### pixel_shuffling

- **PixelShuffleDownscale**
  - methods: forward

### poly_expert_fusion_network

- **MLPProjectionFusion**
  - methods: forward

### polymorphic_activation

- **PolymorphicActivation**: A Polymorphic Activation Function in PyTorch.  This activation function combines aspects of sigmoid and tanh functions, 
  - methods: forward

### polymorphic_neuron

- **PolymorphicNeuronLayer**
  - methods: forward

### prenorm

- **PreNorm**: Prenorm  Args:     dim (_type_): _description_     fn (_type_): _description_
  - methods: forward

### pretrained_t_five

- **PretrainedT5Embedder**
  - methods: run

### proj_then_softmax

- **FusedProjSoftmax**: FusedProjSoftmax is a module that applies a linear projection followed by a softmax operation.  Args:     dim (int): The
  - methods: forward

### pulsar

- **LogGammaActivation**: PulSar Activation function that utilizes factorial calculus  PulSar Activation function is defined as:     f(x) = log(ga
  - methods: forward, backward
- **Pulsar**:     Pulsar Activation function that utilizes factorial calculus      Pulsar Activation function is defined as:         f
  - methods: forward
- **PulsarNew**
  - methods: forward

### pyro

- **hyper_optimize()**: Decorator for PyTorch model optimizations including JIT, FX, Compile, Quantizati

### qformer

- **ImgBlock**: ImgBlock is a module that performs multi-query attention, cross-attention, and feedforward operations on input tensors. 
  - methods: forward
- **TextBlock**: TextBlock module that performs self-attention and feedforward operations.  Args:     dim (int): The dimension of the inp
  - methods: forward
- **QFormer**: QFormer is a transformer-based model for processing text and image inputs.  Args:     dim (int): The dimension of the mo
  - methods: forward
- **img_to_text()**: Convert an image tensor to a text tensor.  Args:     x (Tensor): Input image ten

### qkv_norm

- **qkv_norm()**: Apply QKV normalization.  Args:     q (torch.Tensor): Query tensor.     k (torch
- **qk_norm()**: Apply QK normalization.  Args:     q (torch.Tensor): Query tensor.     k (torch.

### quantized_layernorm

- **QuantizedLN**
  - methods: forward

### query_proposal

- **TextHawkQueryProposal**: A module that represents the TextHawk query proposal model.  Args:     dim (int): The input and output dimension of the 
  - methods: forward

### recurrent_model

- **RNN**: Recurrent Neural Network for MNIST classification.  Usage:     net = RNN(         ntoken=10,         ninp=20,         nh
  - methods: forward

### recursive_block

- **RecursiveBlock**
  - methods: forward

### relu_squared

- **ReluSquared**: Applies the ReLU activation function and squares the output.  Args:     x (torch.Tensor): Input tensor.  Returns:     to
  - methods: forward

### res_net

- **BasicBlock**: BasicBlock   Args:     in_channels (int): Number of input channels     out_channels (int): Number of output channels    
  - methods: forward
- **ResNet**: ResNet  Args:     block (_type_): _description_     num_blocks (_type_): _description_     num_classes (int): Number of 
  - methods: forward

### residual

- **Residual**: Residual connection.  Args:     fn (nn.Module): The module.  Example:     >>> module = Residual(nn.Linear(10, 10))     >
  - methods: forward

### resnet

- **ResNet**: Resnet implementation.  Usage:     from zeta.nn import ResNet      x = torch.randn(1, 3, 224, 224)     net = ResNet()   
- **make_layer()**

### return_loss_text

- **TextTokenEmbedding**
  - methods: forward
- **exists()**
- **return_loss_text()**: Computes the cross-entropy loss between the predicted logits and the target labe
- **add_masking_llm()**: Adds masking to the input tensor.  Args:     x (Tensor): The input tensor.     i
- **calc_z_loss()**
- **max_neg_value()**
- **l2norm()**: Applies L2 normalization to the input tensor.  Args:     x (Tensor): The input t
- **dropout_seq()**: Applies dropout to a sequence of tensors.  Args:     seq (Tensor): The input seq
- **transformer_generate()**: Generates text given a prompt.  Args:     model (nn.Module): The model to genera

### rms_norm

- **RMSNorm**: RMS  Normalization  Args:     dim (int): The dimension of the input.     eps (float): The epsilon value.  Attributes:   
  - methods: forward

### rnn_nlp

- **RNNL**: RNN for language  Args: vocab_size: size of vocabulary embedding_dim: dimension of embedding hidden_dim: dimension of hi
  - methods: forward

### s4

- **s4d_kernel()**: Compute the S4D convolution kernel for state space models on 3D tensors with sha

### scale

- **Scale**: Scale  Args:     value (float): scale value     fn (callable): function to scale   Attributes:     value (float): scale 
  - methods: forward

### scale_norm

- **ScaleNorm**: Applies scale normalization to the input tensor along the last dimension.  Args:     dim (int): The dimension of the inp
  - methods: forward

### scaled_sinusoidal

- **ScaledSinuosidalEmbedding**: scaled sinusoidal embedding  Args:     dim (int): dimension of the embedding  Returns:     torch.Tensor: embedding of sh
  - methods: forward
- **exists()**
- **divisible_by()**

### scalenorm

- **ScaleNorm**: ScaleNorm  Args:     dim (int): dimension of the embedding     eps (float): epsilon value  Attributes:     g (nn.Paramet
  - methods: forward

### shift_tokens

- **ShiftTokens**: Shift Tokens  Overview: Shift tokens in the input sequence  Args:     shifts (list): list of shifts     fn (nn.Module): 
  - methods: forward
- **pad_at_dim()**
- **exists()**
- **shift()**

### shufflenet

- **ShuffleNet**: ShuffleNet implementation.   Usage:     from zeta.nn import ShuffleNet      x = torch.randn(1, 3, 224, 224)     net = Sh
  - methods: forward

### sig_lip

- **NeighbourExchange**
  - methods: forward, backward
- **NeighbourExchangeBidir**
  - methods: forward, backward
- **SigLipLoss**: SigLIP loss module.  Args:     cache_labels (bool, optional): cache labels for faster computation. Defaults to False.   
  - methods: get_ground_truth, get_logits, forward
- **neighbour_exchange()**
- **neighbour_exchange_bidir()**
- **neighbour_exchange_with_grad()**
- **neighbour_exchange_bidir_with_grad()**

### sig_lip_loss

- **SigLipSigmoidLoss**: SigmoidLoss is a custom loss function that computes the sigmoid loss between image and text embeddings.  Args:     dim (
  - methods: forward

### sigmoid_attn

- **SigmoidAttention**: Implements Sigmoid Attention Mechanism.  This replaces the traditional softmax in attention with a sigmoid function. Add
  - methods: forward

### simple_attention

- **simple_attention()**

### simple_feedforward

- **SimpleFeedForward()**: Feedforward neural network with LayerNorms and GELU activations   Flow: layer_no

### simple_lstm

- **SimpleLSTMCell**
  - methods: forward
- **SimpleLSTM**: Simple LSTM implementation.  Args:     dim (int): The input dimension.     hidden_dim (int): The hidden dimension.     d
  - methods: forward

### simple_mamba

- **MambaBlock**: Initialize a single Mamba block.  Args:     dim (int): The input dimension.     dim_inner (Optional[int]): The inner dim
  - methods: forward, ssm, selective_scan
- **Mamba**: Mamba model.  Args:     vocab_size (int): The size of the vocabulary.     dim (int): The input dimension.     depth (int
  - methods: forward

### simple_res_block

- **SimpleResBlock**: Simple residual block with GELU activation  Args:     channels: number of input/output channels  Returns:     x + proj(x
  - methods: forward

### simple_resblock

- **SimpleResBlock**: A simple residual block module.  Args:     channels (int): The number of input and output channels.  Attributes:     pre
  - methods: forward

### simple_rmsnorm

- **SimpleRMSNorm**: SimpleRMSNorm  Args:     dim (int): dimension of the embedding  Usage: We can use SimpleRMSNorm as a layer in a neural n
  - methods: forward

### simple_rnn

- **SimpleRNN**: A simple recurrent neural network module.  Args:     dim (int): The input dimension.     hidden_dim (int): The dimension
  - methods: forward

### skip_connect

- **SkipConnection**
  - methods: forward

### skipconnection

- **SkipConnection**: A helper class to implement skip connections. Adds two input tensors element-wise.  # Example usage from zeta.nn import 
  - methods: forward

### slerp_model_merger

- **SLERPModelMerger**: A class to merge models using Spherical Linear Interpolation (SLERP).  SLERP provides a method to interpolate between tw
  - methods: merge

### snake_act

- **Snake**
  - methods: forward

### sp_act

- **SPAct**
  - methods: forward

### space_time_unet

- **SinusoidalPosEmb**
  - methods: forward
- **RMSNorm**
  - methods: forward
- **GEGLU**
  - methods: forward
- **FeedForwardV**
  - methods: forward
- **ContinuousPositionBias**: from https://arxiv.org/abs/2111.09883
  - methods: device, forward
- **Attention**
  - methods: forward
- **PseudoConv3d**
  - methods: forward
- **SpatioTemporalAttention**
  - methods: forward
- **Block**
  - methods: forward
- **ResnetBlock**
  - methods: forward
- **Downsample**
  - methods: forward
- **Upsample**
  - methods: init_, init_conv_, forward
- **SpaceTimeUnet**
  - methods: forward
- **exists()**
- **default()**
- **mul_reduce()**
- **divisible_by()**
- **shift_token()**

### spacial_transformer

- **SpatialTransformer**: Spacial Transformer Network  https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html  Usage: >>> st
  - methods: stn

### sparc_alignment

- **SparseFineGrainedContrastiveAlignment**
  - methods: forward, global_contrastive_loss, fine_grained_alignment, fine_grained_contrastive_loss

### sparq_attn

- **SparQAttention**: Sparse and Quantized Attention (SparQAttention) is a novel attention mechanism that approximates the attention scores us
  - methods: forward

### sparse_moe

- **GELU_**
  - methods: forward
- **Experts**
  - methods: forward
- **Top2Gating**
  - methods: forward
- **NormalSparseMoE**: NormalSparseMoE is a module that implements the Normal Sparse Mixture of Experts.  Args:     dim (int): The input dimens
  - methods: forward
- **HeirarchicalSparseMoE**
  - methods: forward
- **default()**
- **cast_tuple()**
- **top1()**
- **cumsum_exclusive()**
- **safe_one_hot()**
- **init_()**

### sparse_token_integration

- **SparseTokenIntegration**: SparseTokenIntegration module for integrating sparse tokens into image data.  Args:     dim (int): Dimension of the inpu
  - methods: forward
- **SparseChannelIntegration**: SparseChannelIntegration module integrates sparse tokens into the input image using channel-wise operations.  Args:     
  - methods: forward
- **pair()**

### spatial_downsample

- **SpatialDownsample**: Spatial Downsample Module -------------------------  This module is used to downsample the spatial dimension of a tensor
  - methods: forward
- **exists()**
- **default()**
- **identity()**
- **divisible_by()**
- **pack_one()**
- **unpack_one()**
- **is_odd()**
- **cast_tuple()**

### spatial_transformer

- **SpatialTransformer**: Spacial Transformer Network  https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html  Usage: >>> st
  - methods: stn

### splines

- **B_batch()**: evaludate x on B-spline bases  Args: -----     x : 2D torch.tensor         input
- **coef2curve()**: converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curv
- **curve2coef()**: converting B-spline curves to B-spline coefficients using least squares.  Args: 

### squeeze_excitation

- **SqueezeExcitation**: Squeeze-and-Excitation block.  Parameters --------- in_planes : int     the number of input channels reduced_dim : int  
  - methods: forward

### ssm

- **SSM**
  - methods: forward
- **selective_scan()**: Perform selective scan operation on the input tensor.  Args:     x (torch.Tensor
- **selective_scan_seq()**: Perform selective scan sequence operation on the input tensor.  Args:     x (tor

### ssm_language

- **SSML**: Initialize a single Mamba block.  Args:     dim (int): The input dimension.     dim_inner (Optional[int]): The inner dim
  - methods: forward, ssm, selective_scan

### stoch_depth

- **StochDepth**
  - methods: forward

### stochastic_depth

- **StochasticSkipBlocK**: A module that implements stochastic skip connections in a neural network.  Args:     sb1 (nn.Module): The module to be s
  - methods: forward

### subln

- **SubLN**: SubLN (Subtraction & Layer Normalization) module.  This module computes the subln function: subln(x) = x + fout(LN(fin(L
  - methods: forward

### super_resolution

- **SuperResolutionNet**: Super Resolution Network for MNIST classification.  Usage:     net = SuperResolutionNet()     net(x)
  - methods: forward

### swarmalator

- **pairwise_distances()**
- **function_for_x()**
- **function_for_sigma()**
- **simulate_swarmalators()**:     Swarmalator      Args:         N (int): Number of swarmalators         J (fl

### swiglu

- **SwiGLU**: _summary_  Args:     nn (_type_): _description_
  - methods: forward
- **SwiGLUStacked**: SwiGLUStacked  Args:     nn (_type_): _description_  Examples: >>> from zeta.nn.modules.swiglu import SwiGLUStacked >>> 
  - methods: forward

### tensor

- **Tensor**

### tensor_shape

- **TensorShape**: Represents the shape of a tensor.  Args:     data (array-like): The data of the tensor.     shape_string (str): The stri
  - methods: parse_shape_string, check_shape
- **check_tensor_shape()**: Decorator function that checks if the shape of a tensor matches the specified sh
- **create_tensor()**

### tensor_to_int

- **tensor_to_int()**: Converts a tensor to an integer value based on the specified reduction operation

### text_scene_fusion

- **TextSceneAttentionFusion**: TextSceneAttentionFusion is an attention-based fusion mechanism to combine text sequences and 3D scene embeddings. The m
  - methods: forward

### text_video_fuse

- **TextVideoAttentionFusion**: Text-Video Attention Fusion  Args:     text_features (int): Text features     video_features (int): Video features  Shap
  - methods: forward

### time_up_sample

- **TimeUpSample2x**: Time Up Sample Module  This module is used to upsample the time dimension of a tensor.  Args:     dim (int): The number 
  - methods: init_conv, forward
- **exists()**
- **identity()**
- **divisible_by()**
- **pack_one()**
- **unpack_one()**
- **is_odd()**
- **cast_tuple()**

### to_logits

- **to_logits()**: Converts the input tensor `x` into logits using a sequential layer.  Args:     x

### token_learner

- **TokenLearner**: TokenLearner  TokenLearner is a module that learns tokens from a sequence of tokens.  Args:     dim (int): The input and
  - methods: forward
- **pack_one()**
- **unpack_one()**

### token_mixer

- **TokenMixer()**: TokenMixer module that performs token mixing in a neural network.  Args:     num

### token_shift

- **token_shift()**

### top_n_gating

- **TopNGating**: TopNGating  Args:     dim (int): The input dimension.     num_gates (int): The number of gates.     eps (float, optional
  - methods: forward
- **cast_tuple()**
- **log()**
- **gumbel_noise()**
- **cumsum_exclusive()**
- **safe_one_hot()**

### transformations

- **ResizeMaxSize**
  - methods: forward
- **get_mean_std()**
- **image_transform()**: Image transformations for OpenAI dataset.  Args:     image_size (int): Image siz

### triple_skip

- **TripleSkipBlock**
  - methods: forward

### triton_rmsnorm

- **rms_norm_kernel()**
- **trmsnorm()**: Applies the Triton RMSNorm operation to the given hidden states.  Args:     hidd

### u_mamba

- **UMambaBlock**: UMambaBlock is a 5d Mamba block that can be used as a building block for a 5d visual model From the paper: https://arxiv
  - methods: forward

### unet

- **DoubleConv**
  - methods: forward
- **Down**
  - methods: forward
- **Up**
  - methods: forward
- **OutConv**
  - methods: forward
- **Unet**: UNET model  Flow:     1. Downsample     2. Upsample     3. Output  Args:     n_channels (int): Number of input channels 
  - methods: forward, use_checkpointing

### v_layernorm

- **VLayerNorm**
  - methods: forward

### v_pool

- **DepthWiseConv2d**
  - methods: forward
- **Pool**
  - methods: forward

### video_autoencoder

- **CausalConv3d**: Causal Convolution Module -------------------------  This module is used to perform a causal convolution on a 3D tensor.
  - methods: forward
- **exists()**
- **default()**
- **identity()**
- **divisible_by()**
- **pack_one()**
- **unpack_one()**
- **is_odd()**
- **cast_tuple()**

### video_diffusion_modules

- **TemporalDownsample**: Temporal downsample module that reduces the time dimension of the input tensor by a factor of 2.  Args:     dim (int): T
  - methods: forward
- **TemporalUpsample**: Upsamples the temporal dimension of the input tensor using transposed convolution.  Args:     dim (int): The number of i
  - methods: forward
- **ConvolutionInflationBlock**: Convolution Inflation Block module.  Args:     dim (int): Number of input channels.     conv2d_kernel_size (int): Kernel
  - methods: forward
- **AttentionBasedInflationBlock**: Attention-based inflation block module.  Args:     dim (int): The input dimension.     heads (int): The number of attent
  - methods: forward
- **divisible_by()**
- **exists()**
- **pack_one()**
- **unpack_one()**
- **compact_values()**
- **is_odd()**
- **init_bilinear_kernel_1d()**

### video_to_tensor

- **video_to_tensor()**: Transforms a video file into a PyTorch tensor.  Args:     file_path (str): The p
- **video_to_tensor_vr()**: Transforms a video file into a PyTorch tensor.  Args:     file_path (str): The p

### video_to_text

- **video_to_text()**: Convert a video tensor to a text tensor.  Args:     x (Tensor): Input video tens

### vision_mamba

- **VisionMambaBlock**: VisionMambaBlock is a module that implements the Mamba block from the paper Vision Mamba: Efficient Visual Representatio
  - methods: forward

### vision_weighted_permute_mlp

- **VisionWeightedPermuteMLP**: VisionWeightedPermuteMLP module applies weighted permutation to the input tensor based on its spatial dimensions (height
  - methods: forward

### visual_expert

- **VisualExpert**: Visual Expert from https://arxiv.org/pdf/2311.03079.pdf  Visual expert module. We add a visual expert module to each lay

### vit_denoiser

- **VisionAttention**
  - methods: forward
- **VitTransformerBlock**: Transformer block used in the Vision Transformer (ViT) denoiser model.  Args:     dim (int): The input dimension of the 
  - methods: forward
- **to_patch_embedding()**: Converts the input tensor into patch embeddings.  Args:     x (Tensor): The inpu
- **posemb_sincos_2d()**: Computes positional embeddings using sine and cosine functions for a 2D grid.  A

### vss_block

- **VSSBlock**: VSSBlock is a module that implements a Variational State Space (VSS) block.  PAPER: https://arxiv.org/pdf/2401.10166.pdf
  - methods: forward

### ws_conv2d

- **WSConv2d**: Weight Standardized Convolutional 2D Layer.  This class inherits from `nn.Conv2d` and adds weight standardization to the
  - methods: standardized_weights, forward

### yolo

- **yolo()**: Yolo for object detection  Args:     input: input tensor     num_classes: number

