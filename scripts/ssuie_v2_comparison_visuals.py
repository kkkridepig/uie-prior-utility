"""Add strong-control panels for already-scored, fixed visual cases only."""
import argparse
from pathlib import Path
import torch

from uie_next.records import ROOT, read, write, sha, append
from uie_next.math.utility import labels
from uie_next.v2.context import State
from uie_next.v2.diagnostics import Diagnostics
from uie_next.v2.evaluation import RegistryRuntime
from uie_next.v2.delivery import panel, rgb, heat, selected_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['utility_val', 'sealed_eval'], default='utility_val')
    args = parser.parse_args(); s = State(); role = args.role
    if not s.state.get('closeout_complete') or s.state['scientific_status'] in ['RUNNING', 'INTERRUPTED_RECOVERABLE']:
        raise RuntimeError('Wait for scientific closeout')
    if role == 'sealed_eval' and not s.state['sealed_eval_released']:
        raise PermissionError('Sealed set was not released')
    original = s.run / 'figures' / (role + '_manifest.json')
    if not original.exists():
        raise RuntimeError('No actual fixed cases available')
    selection = read(s.run/'selection/calibration_selection.json')
    registry = read(s.run/'method_registry.json'); primary = selection['primary_control']
    metrics_path = s.run/'metrics/parts'/(role+'_nominal.json')
    scored = read(metrics_path)
    candidate_rows = [r for r in scored['rows'] if r['method'] == 'B1_producer']
    expected_images = 136 if role == 'utility_val' else 177
    if len(candidate_rows) != expected_images or len(scored['done']) != expected_images:
        raise ValueError('Incomplete candidate scores; fixed visual selection denied')
    # Protocol section 15 ranks candidate DeltaPSNR, rather than learned-policy
    # DeltaPSNR. Preserve earlier panels and regenerate both method/control sides.
    case_ids = selected_cases(candidate_rows)
    final_freeze = None
    if role == 'sealed_eval':
        final_freeze = read(s.run/'selection/selection_freeze_before_eval.json')
        if (final_freeze['registry'] != sha(s.run/'method_registry.json')
                or final_freeze['calibration'] != sha(s.run/'selection/calibration_selection.json')):
            raise ValueError('Final identity changed; sealed visualization denied')
    for entry in registry.values():
        for field in ['candidate', 'controller']:
            if entry[field] and sha(entry[field]['path']) != entry[field]['checkpoint_sha256']:
                raise ValueError('Frozen weight changed; visualization denied')
    ident = {'registry': sha(s.run/'method_registry.json'),
             'selection': sha(s.run/'selection/calibration_selection.json'),
             'prior_O_ranked_cases': sha(original), 'candidate_scores': sha(metrics_path),
             'fixed_candidate_case_ids': case_ids, 'script': sha(__file__)}
    receipt = s.run/'figures'/('comparison_'+role+'_manifest.json')
    if receipt.exists():
        if read(receipt)['identity'] != ident:
            raise RuntimeError('Comparison visual identity mismatch')
        print('Reused verified fixed-case comparison visuals'); return
    append(s.run/'commands.jsonl', {'argv': ['-m', 'scripts.ssuie_v2_comparison_visuals', '--role', role],
                                  'purpose': 'already-scored fixed cases; no model/policy/score selection'})
    cases = [{'sample_id': sid} for sid in case_ids]; records = []
    write(s.run/'figures'/('candidate_ranked_'+role+'_selection.json'), {
        'identity': ident, 'sample_ids': case_ids,
        'rule': '8 sample-id SHA256 + candidate DeltaPSNR worst4/best4/middle4; ties sample ID; deduplicate',
        'ranking_method': 'B1_producer', 'uses_reference_for_case_display_only': True,
        'prior_O_ranked_panels_preserved_not_protocol_case_selection': True,
        'impact': 'Case display only; no outputs, labels, quality scores, checkpoints or policies changed.'})
    # load/verify own their device leases; do not nest the repository-wide lock.
    d = Diagnostics(s); d.load(); d.verify_reuse()
    with s.device_job('CLOSEOUT_STRONG_CONTROL_VISUAL_'+role, 300, final=True):
        runtime = RegistryRuntime(d, registry)
        if final_freeze is not None:
            d.data.guard.final_freeze = final_freeze
        with torch.no_grad():
            for case in cases:
                sid = case['sample_id']; b = d.data.batch([sid], 'develop' if role == 'utility_val' else 'final')
                candidates, predictions = runtime.features(b['image'], b['base'])
                bases = {k:v[None].to('cuda:0') for k,v in d.data.base_pair(d.data.by_id[sid]).items()}
                out, alpha = runtime.output('O', selection['methods']['O']['policy'], b['image'], b['base'], candidates, predictions, bases)
                control, _ = runtime.output(primary, selection['methods'][primary]['policy'], b['image'], b['base'], candidates, predictions, bases)
                producer = candidates[registry['O']['candidate']['checkpoint_sha256']]
                truth = labels(b['base'], producer, b['target'])
                delta_error = (b['base']-b['target']).square().mean(1,keepdim=True)-(out-b['target']).square().mean(1,keepdim=True)
                images = [('Input',rgb(b['image'][0])),('Reference (display)',rgb(b['target'][0])),
                          ('J0 clip01',rgb(b['base'][0])),('Producer '+registry['O']['candidate']['checkpoint_id'],rgb(producer[0])),
                          ('Strong control '+primary,rgb(control[0])),('O frozen policy',rgb(out[0])),
                          ('Actual |r| scale .05',heat(truth['r'][0].abs(),.05)),('True U scale .005',heat(truth['U'][0],.005,True)),
                          ('Predicted U scale .005',heat(predictions['O']['U_hat'][0],.005,True)),
                          ('Shared RGB alpha',heat(alpha[0],1)),('MSE improvement scale .005',heat(delta_error[0],.005,True))]
                path=s.run/'figures'/('comparison_'+role)/(sid.replace('/','_')+'.png'); panel(path,images)
                records.append({'sample_id':sid,'path':str(path),'sha256':sha(path),
                    'primary_control':primary,'uses_reference_for_visual_diagnostics':True,
                    'inference_reference_access':False,'candidate_ranked_protocol_cases':True})
    write(receipt,{'identity':ident,'cases':records,'no_new_quality_scores':True,'selection_unchanged':True,
                   'case_selection_basis':'B1_producer DeltaPSNR per protocol section 15'})
    s.live(current_job='none: scientific closeout; strong-control visuals recorded')
    print('Rendered %d fixed comparison panels'%len(records))


if __name__ == '__main__':
    main()
