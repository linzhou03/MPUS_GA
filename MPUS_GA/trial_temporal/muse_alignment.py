"""MUSE controller shares the existing R2 EMA teacher and subgroup bank."""
from dataclasses import asdict
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F

from .multiscale_evidence import (TargetSupervisionTracker, multiscale_evidence,
    target_supervision_loss, muse_ramp, assert_scale_trial_ids, reject_target_truth)
from .subgroup_alignment import trial_keys
from .rejectable_ot import rejectable_transport


class MuseController(nn.Module):
    def __init__(self, model, config, subgroup_config):
        super().__init__()
        self.config = config
        self.warmup, self.ramp_end = subgroup_config.warmup, subgroup_config.ramp_end
        self.tracker = TargetSupervisionTracker(model.num_classes, config).to(next(model.parameters()).device)
        self.evidence = None
        self.current = {}
        self.history = []
        self.iteration = 0
        self.snapshots = [{}, {}]
        self.ot_plans = {}

    @torch.no_grad()
    def observe(self, probability, batch, iteration):
        reject_target_truth(batch)
        assert_scale_trial_ids(batch)
        self.iteration = iteration
        self.evidence = multiscale_evidence(probability, self.config)
        counts = self.tracker.update(self.evidence)
        e = self.evidence
        self.current = {**counts, **self.tracker.summary(iteration,self.warmup,self.ramp_end),
                        'hard_count_total':int(e['hard_mask'].sum()),
                        'partial_count_total':int(e['partial_mask'].sum()),
                        'unsupervised_count_total':int(e['unsupervised_mask'].sum()),
                        'mean_confidence':float(e['confidence'].mean()), 'mean_jsd':float(e['jsd'].mean()),
                        'two_of_three_vote_ratio':float((e['votes']>=2).float().mean()),
                        'three_of_three_vote_ratio':float((e['votes']==probability.shape[1]).float().mean()),
                        'ramp':muse_ramp(iteration,self.warmup,self.ramp_end)}

    def class_gate(self):
        return self.tracker.gate(self.iteration,self.warmup,self.ramp_end)

    @torch.no_grad()
    def observe_target_subgroups(self, bank, keys, features, probability, iteration):
        e=self.evidence
        valid=e['hard_mask'][:,None] & (probability.argmax(-1)==e['hard_label'][:,None])
        bank.observe(1,keys,features,e['hard_label'],e['confidence'],valid,iteration)

    @torch.no_grad()
    def after_bank_refresh(self, bank):
        # Snapshot existing memberships; no additional clustering is performed.
        self.snapshots=deepcopy(bank.memory)
        if self.config.ot_enabled:
            bank.muse_transport=torch.zeros_like(bank.similarity)
            bank.matches.zero_();self.ot_plans={}
            for scale in range(bank.scales):
                for label in range(bank.classes):
                    ns=int((bank.support[0,scale,label]>0).sum());nt=int((bank.support[1,scale,label]>0).sum())
                    source=F.normalize(bank.prototypes[0,scale,label,:ns],dim=-1)
                    target=F.normalize(bank.prototypes[1,scale,label,:nt],dim=-1)
                    cost=(1-source@target.T).clamp(0,2)
                    transport=rejectable_transport(cost,self.config)
                    bank.matches[scale,label,:ns,:nt]=transport['accepted']
                    bank.muse_transport[scale,label,:ns,:nt]=transport['real_mass']
                    self.ot_plans[(scale,label)]=transport['plan'].cpu()
            if bank.history:bank.history[-1]=bank.state()

    def _live_centers(self, bank, domain, scale, label, features, labels, valid, keys):
        count=int((bank.support[domain,scale,label]>0).sum())
        prototypes=bank.prototypes[domain,scale,label,:count]
        if count==0:return prototypes,torch.zeros(0,dtype=torch.bool,device=features.device),torch.zeros(0,dtype=torch.bool,device=features.device)
        current={key:i for i,key in enumerate(keys)}
        groups=[[] for _ in range(count)]
        for key,row in self.snapshots[domain].items():
            if row[1]!=label or not row[3][scale]:continue
            similarity=F.normalize(row[0][scale].to(features),dim=-1)@prototypes.T
            score,slot=similarity.max(0)
            if score>=bank.config.assignment_threshold:groups[int(slot)].append((key,row))
        centers=[];live=[];exists=[]
        for slot,rows in enumerate(groups):
            vectors=[];weights=[];has_current=False
            for key,row in rows:
                index=current.get(key)
                if index is not None:
                    if not valid[index,scale] or labels[index]!=label:continue
                    vector=F.normalize(features[index,scale],dim=-1);has_current=True
                else:vector=row[0][scale].to(features).detach()
                vectors.append(vector);weights.append(row[2])
            if vectors:
                w=features.new_tensor(weights)
                center=(torch.stack(vectors)*w[:,None]).sum(0)/w.sum().clamp_min(self.config.eps)
                centers.append(F.normalize(center,dim=-1));exists.append(True)
            else:centers.append(prototypes[slot].detach());exists.append(False)
            live.append(has_current)
        return torch.stack(centers),torch.tensor(live,device=features.device),torch.tensor(exists,device=features.device)

    def subgroup_loss(self, alignment, source_outputs, source_batches, target_output, target_batch):
        bank=alignment.bank;gate=self.class_gate();e=self.evidence
        sf=torch.cat([o['scale_embeddings'] for o in source_outputs]);tf=target_output['scale_embeddings']
        sy=torch.cat([b['y'].to(sf.device) for b in source_batches])
        sk=[key for d,b in enumerate(source_batches) for key in trial_keys(b,d)];tk=trial_keys(target_batch,0)
        sv=torch.ones(sf.shape[:2],dtype=torch.bool,device=sf.device)
        tv=e['hard_mask'][:,None].expand(tf.shape[:2])
        zero=(sf.sum()+tf.sum())*0;per_class=[]
        for c in range(bank.classes):
            scale_losses=[]
            for s in range(bank.scales):
                ns=int((bank.support[0,s,c]>0).sum());nt=int((bank.support[1,s,c]>0).sum())
                if gate[c]<=0 or not ns or not nt:
                    scale_losses.append(zero);continue
                source,live_s,exists_s=self._live_centers(bank,0,s,c,sf,sy,sv,sk)
                target,live_t,exists_t=self._live_centers(bank,1,s,c,tf,e['hard_label'],tv,tk)
                accepted=bank.matches[s,c,:ns,:nt] & exists_s[:,None] & exists_t[None,:] & (live_s[:,None]|live_t[None,:])
                transport=getattr(bank,'muse_transport',None) if self.config.ot_enabled else None
                mass=(transport[s,c,:ns,:nt] if transport is not None else bank.matches[s,c,:ns,:nt].to(sf)).detach()*accepted
                cost=(1-source@target.T).clamp(0,2)
                scale_losses.append((mass*cost).sum()/(mass.sum()+self.config.eps) if accepted.any() else zero)
            per_class.append(torch.stack(scale_losses).mean())
        values=torch.stack(per_class)
        details={'subgroup_loss_per_class':values.detach().tolist(),
                 'gated_subgroup_loss_per_class':(values.detach()*gate).tolist(),
                 'source_subgroups':(bank.support[0]>0).sum(-1).tolist(),
                 'target_subgroups':(bank.support[1]>0).sum(-1).tolist(),
                 'real_real_matches':bank.matches.sum((-1,-2)).tolist(),
                 'unmatched_source':((bank.support[0]>0)&~bank.matches.any(-1)).sum(-1).tolist(),
                 'unmatched_target':((bank.support[1]>0)&~bank.matches.any(-2)).sum(-1).tolist(),
                 'ot_mean_cost':None}
        if self.config.ot_enabled:
            mass=getattr(bank,'muse_transport',torch.zeros_like(bank.similarity))
            cost=(1-bank.similarity).clamp(0,2)
            details['ot_mean_cost']=float((mass*cost).sum()/mass.sum().clamp_min(self.config.eps))
            details['ot_accepted_mass']=mass.sum((-1,-2)).tolist()
        return (values*gate).sum(),details

    def loss(self, alignment, source_outputs, source_batches, target_output, target_batch, iteration):
        reject_target_truth(target_batch)
        for batch in source_batches:assert_scale_trial_ids(batch)
        if self.evidence is None or self.iteration != iteration:
            raise RuntimeError('MUSE requires current R2 teacher evidence')
        ramp=muse_ramp(iteration,self.warmup,self.ramp_end)
        hard,partial=target_supervision_loss(target_output['scale_logits'],self.evidence,
                                            self.tracker.class_weights(ramp),self.config)
        sub=target_output['scale_logits'].sum()*0
        if ramp == 0:
            hard,partial=sub,sub
        details={'subgroup_loss_per_class':[0.]*self.tracker.M.numel()}
        if self.config.alignment_enabled:
            sub,details=self.subgroup_loss(alignment,source_outputs,source_batches,target_output,target_batch)
        total=ramp*(self.config.lambda_hard*hard+self.config.lambda_partial*partial)+self.config.lambda_subgroup*sub
        record={**self.current, **details, 'L_hard':float(hard.detach()),'L_partial':float(partial.detach()),
                'L_sub':float(sub.detach()),'added_loss':float(total.detach())}
        return total,record

    def record(self, row):
        self.history.append(row)

    def get_extra_state(self):
        return {'iteration':self.iteration,'snapshots':self.snapshots,'ot_plans':self.ot_plans}

    def set_extra_state(self,state):
        self.iteration=state['iteration'];self.snapshots=state['snapshots'];self.ot_plans=state.get('ot_plans',{})

    @torch.no_grad()
    def load_checkpoint(self, state_dict, bank):
        """Restore MUSE state alongside the caller's restored original R2 bank."""
        self.load_state_dict(state_dict)
        if self.config.ot_enabled:
            bank.muse_transport=torch.zeros_like(bank.similarity)
            bank.matches.zero_()
            for (scale,label),plan in self.ot_plans.items():
                ns,nt=plan.shape[0]-1,plan.shape[1]-1
                if ns>bank.config.k or nt>bank.config.k:
                    raise ValueError('Checkpoint subgroup slots differ from the R2 bank')
                mass=plan[:ns,:nt].to(bank.similarity)
                bank.muse_transport[scale,label,:ns,:nt]=mass
                bank.matches[scale,label,:ns,:nt]=mass>0

    def state(self):
        return {'config':asdict(self.config),'teacher':'existing_R2_EMA_shared',
                'probabilities':'unchanged_arithmetic_mean_of_raw_independent_teacher_scales',
                'support_units':'EMA_of_per_batch_counts', 'matching_axes':'independent_scale_then_class',
                'tracker':{k:v.detach().cpu().tolist() for k,v in self.tracker.state_dict().items()},
                'history':self.history}
