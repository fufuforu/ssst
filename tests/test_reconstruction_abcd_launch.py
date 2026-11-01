"""Parse the actual shell-wrapper argv using installed torchrun's parser."""
import contextlib,io,json,shlex,unittest
from pathlib import Path
from torch.distributed.run import get_args_parser
ROOT=Path(__file__).resolve().parents[1]
REPORT=Path('/space/mawb/ssst_reconstruction_abcd_launch_validation_20261025.json')

class LaunchTests(unittest.TestCase):
    def test_actual_wrapper_training_and_export_argv(self):
        lines=[shlex.split(line) for line in (ROOT/'scripts/run_reconstruction_abcd.sh').read_text().splitlines() if '"$PY" -m torch.distributed.run' in line]
        self.assertEqual(len(lines),2);cases=[]
        for tokens in lines:
            for smoke in ([False,True] if 'scripts/train_reconstruction_abcd.py' in tokens else [False]):
                argv=[]
                for token in tokens[3:]:
                    if token=='${args[@]}':argv+=['--smoke'] if smoke else [];continue
                    argv.append(token.replace('$run','/tmp/abcd-launch-check'))
                parsed=get_args_parser().parse_args(argv)
                self.assertTrue((ROOT/parsed.training_script).is_file())
                self.assertEqual(parsed.nproc_per_node,'8')
                if 'train_reconstruction' in parsed.training_script:
                    self.assertEqual(parsed.training_script_args,['--run-root','/tmp/abcd-launch-check']+(['--smoke'] if smoke else []))
                else:self.assertEqual(parsed.training_script_args[:2],['--checkpoint','/tmp/abcd-launch-check/eval/source_checkpoint.pt'])
                cases.append(dict(script=parsed.training_script,arguments=parsed.training_script_args,explicit_separator='--' in argv))
                self.assertIn('--',argv)
        REPORT.write_text(json.dumps(dict(status='PASSED',cases=cases,torchrun_parser='installed torch2.7 get_args_parser',science_changes=[]),indent=2)+'\n')

    def test_previous_ambiguous_argument_reproduces_failure(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            get_args_parser().parse_args(['--standalone','--nproc_per_node=8','scripts/train_reconstruction_abcd.py','--run','/tmp/old','--smoke'])

if __name__=='__main__':unittest.main()
