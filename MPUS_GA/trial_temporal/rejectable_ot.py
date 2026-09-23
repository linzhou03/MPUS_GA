"""Small pure-torch dustbin Sinkhorn; source/target slots are local to one scale."""
import torch


@torch.no_grad()
def rejectable_transport(cost, config):
    if cost.ndim!=2 or not torch.isfinite(cost).all():raise ValueError('OT cost must be finite [Ks,Kt]')
    ns,nt=cost.shape
    full=cost.new_zeros(ns+1,nt+1)
    if not ns or not nt:
        if ns:full[:-1,-1]=1/ns
        if nt:full[-1,:-1]=1/nt
        return {'plan':full,'accepted':torch.zeros_like(cost,dtype=torch.bool),'real_mass':cost.new_zeros(cost.shape)}
    augmented=cost.new_full((ns+1,nt+1),config.ot_null_cost)
    augmented[:ns,:nt]=cost;augmented[-1,-1]=0.
    # Extra capacity permits every real subgroup to choose NULL.
    a=cost.new_ones(ns+1);a[-1]=nt;a/=ns+nt
    b=cost.new_ones(nt+1);b[-1]=ns;b/=ns+nt
    log_kernel=-augmented/config.ot_sinkhorn_epsilon
    u=torch.zeros_like(a);v=torch.zeros_like(b)
    for _ in range(config.ot_sinkhorn_iters):
        u=a.log()-torch.logsumexp(log_kernel+v[None],dim=1)
        v=b.log()-torch.logsumexp(log_kernel+u[:,None],dim=0)
    full=(log_kernel+u[:,None]+v[None]).exp()
    real=full[:ns,:nt].clone()
    accepted=(cost<config.ot_match_cost_threshold)&(real>config.ot_mass_threshold)
    # Rejected real-real flow is explicitly rerouted through NULL.
    rejected=real*~accepted
    full[:ns,:nt]=real*accepted
    full[:ns,-1]+=rejected.sum(1)
    full[-1,:nt]+=rejected.sum(0)
    full[-1,-1]=(full[-1,-1]-rejected.sum()).clamp_min(0)
    return {'plan':full,'accepted':accepted,'real_mass':full[:ns,:nt].clone()}
