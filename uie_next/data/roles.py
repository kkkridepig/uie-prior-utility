import json
from pathlib import Path


ALLOWED = {
    'candidate_train': {'model_fit'}, 'candidate_select': {'model_val'},
    'utility_train': {'utility_fit'}, 'label_scales': {'utility_fit'},
    'matched_train': {'model_fit','utility_fit'}, 'utility_select': {'utility_val'},
    'candidate_diagnostic': {'utility_val'}, 'develop': {'utility_val'},
    'calibrate': {'calibration'}, 'final': {'sealed_eval'},
}


class RoleGuard:
    def __init__(self, rows, final_freeze=None):
        self.rows={r['sample_id']:r for r in rows}
        if len(self.rows)!=len(rows):raise ValueError('duplicate role sample IDs')
        self.final_freeze=final_freeze

    def check(self, sample_id, operation):
        if operation not in ALLOWED: raise PermissionError('unknown data operation')
        row=self.rows[sample_id]
        if row['role'] not in ALLOWED[operation]: raise PermissionError('role forbidden for '+operation)
        if row['role'].startswith('utility') or row['role'] in ('calibration','sealed_eval'):
            if row['upstream_exposure']!='documented_nonoverlap':
                raise PermissionError('unknown or overlapping upstream exposure is not independent')
            if not row.get('backbone_provenance_verified',False):
                raise PermissionError('downloaded backbone provenance not yet verified')
        if operation=='final':
            if not self.final_freeze or self.final_freeze.get('development_status')!='DEV_PASS':
                raise PermissionError('sealed reference not released')
        return row


def read_roles(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
