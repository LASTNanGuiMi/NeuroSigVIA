"""Replace only routing weights while preserving the original v5 host code.

The original source is untouched. A process-local constructor registration lets
the unmodified trainer create this subclass and serialize the explicit policy.
Graph rendering, feature extraction, Q/V processing, value adapters, alignment,
fusion and classifier remain the original code. Shared v5 regularizers are used
for all three policies: this is a controlled selector-mechanism adaptation.
"""
import hashlib
from types import MethodType
import torch
from src import patch_mindts as upstream

OriginalFusionModule=upstream.PatchMindTSFusionModule
DEFAULT_POLICY='current'

def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def selector_distribution(self,projected_query,relative_keys,graph_tokens,
                          patch_mask=None,valid_fraction=None):
    key=str(projected_query.device)
    if key not in self._selector_generators:
        self._selector_generators[key]=torch.Generator(device=projected_query.device).manual_seed(self._selector_noise_seed)
    result=self.selector_policy_module(projected_query,relative_keys,patch_mask=patch_mask,
        valid_fraction=valid_fraction,generator=self._selector_generators[key])
    weights=result['weights']; clean=result['clean_scores']; probabilities=result['pre_topk_weights']
    zero=weights.new_zeros(weights.shape[:2]); scalar_zero=weights.new_zeros(())
    top=probabilities.topk(2,dim=-1).values; margin=top[...,0]-top[...,1]
    diagnostics={
        'dataset_prior_weights':torch.full_like(weights,1./self.num_granularities),
        # These policies are patch-local. Global entries are explicit neutral
        # diagnostics, not reused old-v5 routes or a second stochastic draw.
        'sample_global_weights':torch.full_like(weights,1./self.num_granularities),
        'patch_local_weights':weights,'candidate_evidence':zero,
        'global_mix':scalar_zero,'local_mix':scalar_zero+1,
        'sample_global_score_margin':zero,'patch_local_score_margin':margin,
        'sample_global_confidence':zero,'patch_local_confidence':zero,
        'effective_global_mix':zero,'effective_local_mix':zero+1,
        'sample_global_route_scores':torch.zeros_like(clean),'patch_local_route_scores':clean,
        'router_pre_topk_scores':clean,'router_pre_topk_weights':probabilities,
        'router_topk_mask':result['topk_mask'],'router_topk_indices':result['topk_indices'],
        'router_noisy_topk_scores':result['route_logits'],'router_noise_std':result['noise_std']}
    return weights.clamp_min(1e-12).log(),weights,weights.unsqueeze(2).expand(-1,-1,self.num_heads,-1),diagnostics

class SelectorComparisonModel(OriginalFusionModule):
    def __init__(self,*args,selector_policy=None,selector_noise_seed=None,**kwargs):
        policy=DEFAULT_POLICY if selector_policy is None else selector_policy
        if policy not in {'current','timemosaic','pathformer'}: raise ValueError('Unknown selector policy')
        if kwargs.get('granularity_router_mode','adaptive_v5')!='adaptive_v5':
            raise ValueError('Selector comparison retains the v5 representation host')
        kwargs['granularity_router_mode']='adaptive_v5'
        kwargs['router_top_k']=1 if policy=='timemosaic' else 2
        super().__init__(*args,**kwargs)
        self.selector_policy=policy
        self.initial_common_state_sha256=state_digest(self.state_dict())
        noise_seed=int(torch.initial_seed()+17071 if selector_noise_seed is None else selector_noise_seed)
        if policy!='current':
            from selector_policies.timemosaic import TimeMosaicSelector
            from selector_policies.pathformer import PathformerSelector
            selector_type=TimeMosaicSelector if policy=='timemosaic' else PathformerSelector
            # Construct AFTER all common layers; restore CPU RNG so shared
            # dropout/DataLoader streams are independent of parameter count.
            with torch.random.fork_rng(devices=[]):
                selector=selector_type(self.granularity_attention.fusion_dim,self.granularity_attention.num_granularities)
            attention=self.granularity_attention
            attention.selector_policy_module=selector
            attention._selector_noise_seed=noise_seed
            attention._selector_generators={}
            attention._routing_distribution=MethodType(selector_distribution,attention)
            # These old scoring-only parameters are unused by a query-input
            # replacement. Preserve their tensors, exclude them from training.
            attention.v5_relation_scorer.requires_grad_(False)
            attention.key_projection.requires_grad_(False)
            attention.v5_key_adapter_delta.requires_grad_(False)
        self.constructor_configuration.update(selector_policy=policy,selector_noise_seed=noise_seed)
        self.configuration.update(self.constructor_configuration)
        self.configuration.update(
            selector_comparison_scope='routing_weights_only_original_images_and_downstream',
            initial_common_state_sha256=self.initial_common_state_sha256,
            selector_initialization_rng='isolated_after_all_common_layers',
            selector_noise_rng='private_generator_does_not_consume_dropout_or_loader_rng',
            selector_checkpoint_scope='best_model_inference_not_mid_training_rng_resume',
            selector_training_objective='same_classification_alignment_v5_budget_and_load_for_all_methods')
        if policy!='current':
            self.configuration.update(
                granularity_score=f'{policy}_official_selection_mechanism_projected_query_adapter',
                granularity_router_inputs='projected_line_query_only',
                granularity_decision_scope='one_selection_distribution_per_valid_outer_window',
                granularity_global_pooling='none_patch_local_selector',
                granularity_standalone_scale_bias=True,
                granularity_topk_policy='hard_gumbel_tau0.5_top1' if policy=='timemosaic' else 'learned_noisy_top2_selected_softmax',
                granularity_router_noise_policy='training_only_gumbel' if policy=='timemosaic' else 'training_only_learned_gaussian',
                granularity_regularization_policy='shared_v5_budget_post_weights_and_load_clean_probabilities')

def install_selector_host(policy):
    global DEFAULT_POLICY
    if policy not in {'current','timemosaic','pathformer'}: raise ValueError('Unknown selector policy')
    DEFAULT_POLICY=policy
    upstream.PatchMindTSFusionModule=SelectorComparisonModel
    return SelectorComparisonModel
