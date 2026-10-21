import re
from pathlib import Path

p = Path('/space/mawb/ssst_sem_res256_code_20261019/tokengs/models/unified_query_v1.py')
t = p.read_text()
t = t.replace("if _os.environ.get('RG_EXTRA'):",
              "if _os.environ.get('RG_EXTRA') not in (None, '', '0'):")
t = t.replace("            _extra = 0.0\n",
              "            _extra = torch.zeros((), device=prediction['states'][-1]['tokens'].device)\n")
t = t.replace("                _dl = 0.0\n", "                _dl = torch.zeros((), device=_dp.device)\n")
p.write_text(t)
print('patched', t.count('not in (None'), 'flag guards;', t.count('torch.zeros((), device='), 'tensor inits')
