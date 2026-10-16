"""Build a self-contained server installer from the checked sky-position files."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import textwrap
root=Path(__file__).resolve().parents[1]
files=['gwak/sky_position_antenna_patterns_math.pdf','scripts/build_sky_antenna_math_pdf.py','gwak/sky_position.py','gwak/sky_position_antenna_patterns.py','gwak/sky_position_first_principles.py','scripts/patch_sky_sampler.py','scripts/install_sky_position.py',
       'scripts/verify_sky_position.py','scripts/requirements-sky.txt']
files += ['gwak/data/fixed_sine_gaussian.py']
files += ['examples/sky_position/README.txt', 'examples/sky_position/icecube_event.csv', 'examples/sky_position/sg150_directional_event.json', 'examples/sky_position/IC110514A.png', 'examples/sky_position/IC110514A.pdf', 'examples/sky_position/IC110514A_antenna.png', 'examples/sky_position/IC110514A_antenna.pdf', 'examples/sky_position/run_example.py', 'examples/sky_position/train_event.sh']
manifest={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in files}
(root/'payload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
    for rel in files+['payload_manifest.json']:
        data=(root/rel).read_bytes();info=tarfile.TarInfo(rel);info.size=len(data);info.mode=0o644
        tar.addfile(info,io.BytesIO(data))
raw=buf.getvalue();encoded='\n'.join(textwrap.wrap(base64.b64encode(raw).decode(),100))
script='''#!/bin/bash
set -euo pipefail
SKY_ROOT="${1:-$HOME/gwak-full-run}"
if [ "$#" -gt 0 ]; then shift; fi
SKY_PAYLOAD=$(mktemp -d "${TMPDIR:-/tmp}/gwak-sky-update.XXXXXXXX")
export SKY_PAYLOAD
trap 'rm -rf -- "$SKY_PAYLOAD"' EXIT
python3 - <<'SKY_UNPACK'
import base64,hashlib,io,os,tarfile
from pathlib import Path
root=Path(os.environ['SKY_PAYLOAD'])
raw=base64.b64decode("""ENCODED""")
assert hashlib.sha256(raw).hexdigest() == 'DIGEST', 'Payload checksum mismatch'
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
    for member in tar.getmembers():
        path=Path(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
        dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(tar.extractfile(member).read())
SKY_UNPACK
python3 "$SKY_PAYLOAD/scripts/install_sky_position.py" --root "$SKY_ROOT" "$@"
'''.replace('ENCODED',encoded).replace('DIGEST',hashlib.sha256(raw).hexdigest())
(root/'scripts/install_sky_position.sh').write_text(script)
print('Installer bytes:',len(script))
