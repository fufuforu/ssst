"""One Adam update from the same eight ordered singleton windows on four ranks."""
def sample_indices(order,step,rank,world,accumulation):
    assert world*accumulation==8 and 0<=rank<world
    begin=step*8
    assert begin+8<=len(order)
    return [order[begin+micro*world+rank] for micro in range(accumulation)]


def accumulate(ddp,batches,step,accumulation,*,first_batch=None,after_backward=None):
    totals={};seen=0
    for micro,batch in enumerate(batches):
        assert micro<accumulation
        if micro==0 and first_batch is not None:first_batch(batch)
        # Synchronize every microbatch: existing accumulated grads are already
        # rank-equal. Linearity preserves the global mean and bucket views avoid
        # an extra full gradient allocation from a no_sync first microbatch.
        loss,metrics,lr,lu=ddp(batch,step,audit=micro==0)
        (loss/accumulation).backward()
        for key,value in metrics.items():totals[key]=totals.get(key,0.)+value/accumulation
        del loss,lr,lu,batch
        if after_backward is not None:after_backward()
        seen+=1
    assert seen==accumulation
    return totals
