import torch
import pytest
from scripts.cde_v3.train_bank import select_batch,batch_random,mode_at
from scripts.cde_v3.common import roles,MODES,generator
from scripts.cde_v3.train_selector import choose
from scripts.cde_v3.model import Selector

def test_actual_streams_pair_across_branches():
    rows=roles('adapter_fit'); seen=[]
    for branch in ('BASE_CONT_V3','C_BANK','C_RGB_CONTROL','C_ALL_ONLY'):
        records=[]
        for step in range(12):
            torch.rand(step+17); mode_at(20261004,step,branch)
            rr=select_batch(rows,20261004,step); ids=[r['sample_id'] for r in rr]
            t,n=batch_random((4,3,8,8),20261004,step,ids,'cpu'); records.append((ids,t,n))
        seen.append(records)
    for other in seen[1:]:
        for a,b in zip(seen[0],other):
            assert a[0]==b[0] and torch.equal(a[1],b[1]) and torch.equal(a[2],b[2])
    rr=select_batch(rows,20261005,0)
    assert [r['sample_id'] for r in rr]!=seen[0][0][0]

def test_modes_exactly_balanced_no_null():
    modes=[mode_at(20261004,step,'C_BANK') for step in range(5000)]
    assert all(modes.count(m)==1250 for m in MODES[1:])

def test_policy_units_masks_and_ties():
    stats=torch.ones(1,4); scores=torch.tensor([[0.,.08,-.3,.06,.04]])
    assert choose(scores,stats).item()==1
    assert choose(scores,stats,.1).item()==0
    with pytest.raises(ValueError): choose(scores,stats,.1,'WINNER_CE')
    assert choose(torch.zeros(1,5),stats).item()==0
    assert choose(scores,torch.zeros_like(stats)).item()==0

def test_actual_selector_parameters_are_shared_control_capacity():
    m=Selector(); assert m.head[0].in_features==116 and m.head[-1].out_features==4
