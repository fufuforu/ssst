import numpy as np
import pytest

from scripts.eval_object_locus_frozen_probe import flatten_aligned_batch


def test_ordered_batched_flatten_partial_batches_and_2d_compatibility():
    b,q,c=2,100,19
    p=np.zeros((b,q,c),np.float32)
    for wi in range(b):
        for qi in range(q):p[wi,qi,(wi*q+qi)%c]=1
    y=np.full((b,q),-1,np.int64)
    y[:,::3]=18
    y[:,1::3]=np.arange(66).reshape(2,33)%18
    expected=flatten_aligned_batch(p,y)
    partial_batch=flatten_aligned_batch(p[:2],y[:2]) # capacity 3, final batch has 2 windows
    partial=[flatten_aligned_batch(p[i:i+1],y[i:i+1]) for i in range(2)]
    assert np.array_equal(expected[0],partial_batch[0])
    assert np.array_equal(expected[1],partial_batch[1])
    assert np.array_equal(expected[0],np.concatenate([x[0] for x in partial]))
    assert np.array_equal(expected[1],np.concatenate([x[1] for x in partial]))
    for k in (0,1,99,100,199):
        wi,qi=divmod(k,100)
        assert np.array_equal(expected[0][k],p[wi,qi])
        assert expected[1][k]==y[wi,qi]
    p2,y2=flatten_aligned_batch(p[0],y[0])
    assert np.array_equal(p2,p[0]) and np.array_equal(y2,y[0])
    with pytest.raises(ValueError):flatten_aligned_batch(p,y[:,:-1])
