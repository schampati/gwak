"""Install a reviewed sky-position update in ~/gwak-full-run after CPU preflight.

This does not launch or stop any workers. Existing running Python processes
retain their imported code; use the new interface in future jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from patch_sky_sampler import patch_dataloader


def atomic_write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data)
        if path.exists():os.chmod(tmp,path.stat().st_mode & 0o777)
        else:os.chmod(tmp,0o644)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'gwak-full-run')
    p.add_argument('--image',type=Path,default=Path.home()/'apptainer-images/pytorch-2.3.1-cuda12.1.sif')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();root=args.root.expanduser().resolve()
    payload=Path(__file__).resolve().parents[1]
    dest=root/'gwak/train/dataloader.py'
    if not dest.is_file():raise SystemExit(f'No GWAK dataloader at {dest}')
    manifest=json.loads((payload/'payload_manifest.json').read_text())
    for name,want in manifest.items():
        if hashlib.sha256((payload/name).read_bytes()).hexdigest()!=want:
            raise SystemExit(f'Payload checksum mismatch: {name}')
    # Compute and compile all changes before writing into the checkout.
    original=dest.read_bytes();patched=patch_dataloader(original)
    changes={'gwak/train/dataloader.py':patched}
    for rel in manifest:
        if rel.startswith(('gwak/data/fixed_sine_gaussian.py','gwak/sky_position_antenna_patterns_math.pdf','scripts/build_sky_antenna_math_pdf.py','gwak/sky_position.py','gwak/sky_position_antenna_patterns.py','gwak/sky_position_first_principles.py','examples/sky_position/','scripts/verify_sky_position.py','scripts/requirements-sky.txt')):
            changes[rel]=(payload/rel).read_bytes()
    active_config = root/'gwak/train/configs/ResNet_O4b20_grb221009a.yaml'
    if active_config.exists():
        current = active_config.read_bytes()
        # Preserve numeric geometry and all other settings in the active example.
        changes[str(active_config.relative_to(root))] = current.replace(
            b'fixed_ra:', b'fixed_phi_rad:').replace(b'fixed_dec:', b'fixed_declination_rad:')
    for rel,data in changes.items():
        if rel.endswith('.py'):compile(data,rel,'exec')
    if not shutil.which('apptainer'):raise SystemExit('apptainer is required for the preflight')
    if not args.image.is_file():raise SystemExit(f'Container missing: {args.image}')
    before = {rel: (root/rel).read_bytes() if (root/rel).exists() else None for rel in changes}
    # Home scratch is visible inside the container without any shared /tmp assumption.
    stage=Path(tempfile.mkdtemp(prefix='sky-position-preflight-',dir=root))
    for rel,data in changes.items():
        path=stage/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    paths=[str(stage),str(root/'site-packages'),str(root),str(root/'ml4gw'),str(root/'gwak/train')]
    command=['apptainer','exec','--cleanenv','--bind',str(root)+':'+str(root),
             '--env','PYTHONPATH='+':'.join(paths),'--env','CUDA_VISIBLE_DEVICES=',
             '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
             '--env','OPENBLAS_NUM_THREADS=1','--env','NUMEXPR_NUM_THREADS=1',
             '--env','GOMAXPROCS=2',str(args.image),'python',
             str(stage/'scripts/verify_sky_position.py'),'--root',str(root),'--candidate-dir',str(stage)]
    env=os.environ.copy();env.update(GOMAXPROCS='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    print('CPU preflight; original source is unchanged.',flush=True)
    subprocess.run(command,env=env,check=True,timeout=240)
    if args.check_only:
        print('CHECK ONLY PASSED. Source unchanged. Candidate:',stage);return
    for rel, data in before.items():
        current = (root/rel).read_bytes() if (root/rel).exists() else None
        if current != data:raise SystemExit(f'{rel} changed during preflight; retry after inspecting it')
    backup=root/f'sky-position-backup-{time.time_ns()}';backup.mkdir()
    prior={}
    for rel in changes:
        path=root/rel;prior[rel]=path.read_bytes() if path.exists() else None
        if path.exists():
            saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,saved)
    (backup/'manifest.json').write_text(json.dumps({r:None if d is None else hashlib.sha256(d).hexdigest()
                                                  for r,d in prior.items()},indent=2))
    written=[]
    try:
        for rel,data in changes.items():
            atomic_write(root/rel,data);written.append(rel)
        for rel,data in changes.items():
            assert (root/rel).read_bytes()==data
    except BaseException:
        for rel in reversed(written):
            if prior[rel] is None:(root/rel).unlink(missing_ok=True)
            else:atomic_write(root/rel,prior[rel])
        raise
    (backup/'installed_hashes.json').write_text(json.dumps({r:hashlib.sha256(d).hexdigest()
                                                          for r,d in changes.items()},indent=2))
    print('SKY POSITION UPDATE INSTALLED:',root)
    print('Backup:',backup)
    print('New jobs accept event_ra_deg, event_dec_deg, event_time_utc; default sky_window_deg=1.0.')
    print('No training started; running workers and archived study configurations were not changed.')


if __name__=='__main__':main()
