import copy
import pytest
import yaml
from uie_next.records import ROOT
from uie_next.schema import load


def test_frozen_recipe_and_unknown_keys(tmp_path):
    path=ROOT/'configs/uie_next/protocol.yaml';config=load(path)
    for section,key,value in [('candidate','width',64),('utility','shared_odd_difference',False),('prior','rates',[1,1,1]),('budget','max_device_hours',17)]:
        altered=copy.deepcopy(config);altered[section][key]=value
        output=tmp_path/'protocol.yaml';output.write_text(yaml.safe_dump(altered))
        with pytest.raises(ValueError):load(output)
    config['candidate']['extra']='ignored?'
    path=tmp_path/'unknown.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):load(path)
