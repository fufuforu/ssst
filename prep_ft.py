import json
import re
from pathlib import Path

SNAP = Path('/space/mawb/ssst_sem_res256_code_20261019')
SRC = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/attempt01_A_sem256_full_node17_8gpu')
ROOT = Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/finetune_extra_v1')

PATCH = '''
        import os as _os
        if _os.environ.get('RG_EXTRA'):
            _extra = 0.0
            if _os.environ.get('RG_DEPTH'):
                _st = prediction['states'][-1]
                _g = self.activation_head(_st['tokens'], _st['mu'], _st['radii'])
                _rec = self._reconstruction_from_gaussians(_g)
                _r = self.render_reconstruction(_rec, decoder)
                _dp = _r.get('depths_pred', _r.get('depth'))
                if _dp is None:
                    raise RuntimeError('depth render key missing')
                _dg = batch['depth_gt_m_all'].reshape(_dp.shape)
                _dv = batch['depth_gt_valid_all'].reshape(_dp.shape)
                _dl = 0.0
                _n = 0
                for _v in range(_dp.shape[1]):
                    _p = _dp[0, _v, 0]; _t = _dg[0, _v, 0]; _m = _dv[0, _v, 0]
                    if int(_m.sum()) < 16:
                        continue
                    _x = _p[_m]; _y = _t[_m]
                    _a = (_x * _y).mean() - _x.mean() * _y.mean()
                    _b = (_x * _x).mean() - _x.mean() ** 2 + 1e-8
                    _s = _a / _b
                    _o = _y.mean() - _s * _x.mean()
                    _dl = _dl + (_s * _x + _o - _y).abs().mean()
                    _n += 1
                _extra = _extra + (1.0 * _dl / max(_n, 1))
                metrics['loss_extra_depth'] = float(_extra.detach())
            if _os.environ.get('RG_TSEM'):
                _cal = prediction['target_camera_calibration']
                _d6 = ModelInputDecoder(cam_view=torch.linalg.inv(_cal['c2w']).transpose(-1, -2),
                                        intrinsics=_cal['intrinsics'])
                _out = self._render_understanding(prediction, _d6)
                _sc = _out['semantic_scores']
                _lab = batch['semantic_label_all']
                _v = (_lab >= 0) & (_lab < 20)
                _ce = torch.nn.functional.cross_entropy(_sc[:, 2:].permute(0, 1, 3, 4, 2).reshape(-1, 20)[_v[:, 2:].reshape(-1)],
                                                        _lab[:, 2:].reshape(-1)[_v[:, 2:].reshape(-1)])
                _extra = _extra + 0.1 * _ce
                metrics['loss_extra_tsem'] = float(_ce.detach())
            metrics['loss_extra'] = float(_extra.detach())
            metrics['loss_total_with_extra'] = float((metrics['loss'] + _extra).detach())
            metrics['loss'] = metrics['loss'] + _extra
'''

def patch_step_loss():
    p = SNAP / 'tokengs/models/unified_query_v1.py'
    text = p.read_text()
    if 'RG_EXTRA' in text:
        print('already patched')
        return
    marker = "        prediction['target_camera_calibration']=calibration\n"
    if marker not in text:
        raise SystemExit('marker not found')
    text = text.replace(marker, marker + PATCH)
    p.write_text(text)
    print('patched step_loss')


def pseudo_zero_step(name):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    ck = __import__('torch').load(SRC / 'checkpoint_latest.pt', map_location='cpu', weights_only=False, mmap=True)
    blob = dict(model=ck['model'], optimizer=dict(state={}, param_groups=[]),
                progress=dict(epoch=0, update_in_epoch=0, full_updates=0),
                rank_rng=ck['rank_rng'], meta=ck['meta'])
    __import__('torch').save(blob, out / 'checkpoint_untrained_initial.pt')
    import hashlib
    h = hashlib.sha256()
    with open(out / 'checkpoint_untrained_initial.pt', 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    (out / 'untrained_initial.sha256').write_text(h.hexdigest() + '\n')
    print(name, 'source_updates=4172 -> pseudo zero-step', h.hexdigest()[:16])


if __name__ == '__main__':
    patch_step_loss()
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ('C', 'R1', 'U1'):
        pseudo_zero_step(name)
