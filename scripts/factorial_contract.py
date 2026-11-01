"""Structural autograd audit without an additional backward or retained gradients."""
def leaf_ids(loss):
    pending=[loss.grad_fn];seen=set();leaves=set()
    while pending:
        node=pending.pop()
        if node is None or node in seen:continue
        seen.add(node)
        variable=getattr(node,'variable',None)
        if variable is not None:leaves.add(id(variable))
        pending.extend(edge[0] for edge in node.next_functions)
    return leaves,len(seen)

def graph_isolation(lr,lu,reconstruction,understanding):
    rec={id(p) for p in reconstruction if p.requires_grad};under={id(p) for p in understanding if p.requires_grad}
    assert not rec&under
    r,rnodes=leaf_ids(lr);u,unodes=leaf_ids(lu)
    assert r&rec and u&under,'Loss disconnected from its own trainable branch'
    assert not r&under and not u&rec,'Cross-branch autograd edge'
    return dict(status='PASSED',method='autograd_graph_reachability',LU_to_reconstruction_zero=True,
                LR_to_understanding_zero=True,reconstruction_graph_nodes=rnodes,understanding_graph_nodes=unodes)
