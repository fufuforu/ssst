"""Legacy dual-stream equations with explicit variable superpoint counts."""
import math
import torch
import torch.nn.functional as F
from tokengs.models.object_locus_panoptic_v1_controller import ObjectLocusPanopticV1Controller,RegisteredObjectLayer,geometry_bias,capped_injection


class DynamicObjectLayer(RegisteredObjectLayer):
    def update_geometry(self,R,mu,q,c,s,ell):
        weights=R[:,:c.shape[1]];ce=torch.einsum('bqt,btd->bqd',weights,mu)
        var=torch.einsum('bqt,btqd->bqd',weights,(mu[:,:,None]-ce[:,None]).square())
        low=.05*ell[:,None,None];high=2*ell[:,None,None]
        se=(var+low.square()).sqrt().clamp(min=low,max=high)
        delta=.25*(ce-c)+.05*ell[:,None,None]*self.W_c(self.ln_geom(q)).tanh()
        delta=delta*(.25*ell[:,None,None]/(delta.norm(dim=-1,keepdim=True)+1e-6)).clamp(max=1)
        v=torch.log(s/ell[:,None,None])+.25*torch.log(se/s).clamp(-math.log(2),math.log(2))+.1*self.W_s(self.ln_geom(q)).tanh()
        return c+delta,ell[:,None,None]*v.clamp(math.log(.05),math.log(2)).exp()

    def _read_anchor(self,q,a,bias):
        return self.anchor_attention(q,a,bias)

    def _read_image(self,q,image):
        return self.image_attention(q,self.image_ln(image))

    def forward(self,h,a,mu,q,c,s,ell,image,mask_embedder,f_anchor,exposure,**kwargs):
        if kwargs:raise ValueError('Legacy diagnostic overrides are not supported in this explicit variant')
        ev,R=self._read_anchor(q,a,geometry_bias(mu,c,s));q1=self.norms[0](q+ev)
        im,_=self._read_image(q1,image);q2=self.norms[1](q1+im)
        q3=self.norms[2](q2+self.self_attn(q2,q2,q2,need_weights=False)[0]);qnew=self.norms[3](q3+self.ffn(q3))
        cnew,snew=self.update_geometry(R.mean(1),mu,qnew[:,:c.shape[1]],c,s,ell)
        mq=mask_embedder(qnew)
        logits=F.normalize(f_anchor,dim=-1,eps=1e-6)@F.normalize(mq,dim=-1,eps=1e-6).transpose(1,2)/.1
        route=torch.cat((logits+geometry_bias(mu,cnew,snew).transpose(1,2),logits.new_zeros((*logits.shape[:2],1))),-1).softmax(-1)
        message=route[...,:qnew.shape[1]]@F.layer_norm(qnew,(256,),eps=1e-5)
        return dict(q=qnew,c=cnew,s=snew,route=route,joint_delta=capped_injection(h,message,self.W_inject,exposure),
            joint_h_norm=h.norm(dim=-1),evidence_attention=R,anchor_embedding=a,f_anchor=f_anchor,m_query=mq,
            anchor_mask_logits=f_anchor@mq.transpose(1,2),anchor_membership=(f_anchor@mq.transpose(1,2)).sigmoid(),ell=ell)


class DynamicPanopticController(ObjectLocusPanopticV1Controller):
    def __init__(self):
        super().__init__()
        for layer in self.layers.values():layer.__class__=DynamicObjectLayer

    def gaussian_child_features(self,a,f_anchor,gaussians,mu,radii):
        batch,anchors=a.shape[:2]
        if gaussians.shape!=(batch,anchors*64,14):raise ValueError('64 children per reconstruction superpoint required')
        child=gaussians.reshape(batch,anchors,64,14);radius=radii[:,:,None,None]
        delta=((child[...,:3]-mu[:,:,None])/(radius+1e-6)).clamp(-5,5)
        log_scale=(child[...,4:7].clamp_min(1e-8)/(radius+1e-6)).log().clamp(-5,5)
        geometry=torch.cat((delta,log_scale,child[...,7:11],child[...,3:4],child[...,11:14]),-1)
        emb=self.child_index_embedding.weight[None,None].expand(batch,anchors,-1,-1)
        residual=self.child_mlp(torch.cat((a[:,:,None].expand(-1,-1,64,-1),geometry,emb),-1))
        return (f_anchor[:,:,None]+residual).reshape(batch,anchors*64,256),residual

    def classify(self,q):
        logits=self.class_head(self.class_ln(q[:,:-2]));return dict(thing_logits19=logits,p_class=logits.softmax(-1))
