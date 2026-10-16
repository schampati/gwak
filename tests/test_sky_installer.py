"""Installer transaction checks; fake Apptainer isolates install control flow.

Real module/projection checks run separately with verify_sky_position.py.
"""
import os
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module', autouse=True)
def build_installer():
    import sys
    subprocess.run([sys.executable, str(ROOT/'scripts/package_sky_update.py')], check=True)

@pytest.mark.parametrize('success',[True,False])
def test_installer_preserves_or_backs_up_source(tmp_path,success):
    original=ROOT/'gwak/train/dataloader.py'
    checkout=tmp_path/'checkout';target=checkout/'gwak/train/dataloader.py'
    target.parent.mkdir(parents=True)
    initial=original.read_bytes()+b'\n# An unrelated remote change must survive\n'
    target.write_bytes(initial)
    fakebin=tmp_path/'bin';fakebin.mkdir()
    apptainer=fakebin/'apptainer';apptainer.write_text('#!/bin/sh\nexit '+('0' if success else '19')+'\n')
    apptainer.chmod(0o755)
    image=tmp_path/'image.sif';image.touch()
    env=os.environ.copy();env['PATH']=str(fakebin)+':'+env['PATH']
    result=subprocess.run(['bash',str(ROOT/'scripts/install_sky_position.sh'),str(checkout),'--image',str(image)],
                          env=env,capture_output=True,text=True)
    if success:
        assert result.returncode==0,result.stdout+result.stderr
        assert b'event_time_utc' in target.read_bytes()
        assert target.read_bytes().endswith(b'# An unrelated remote change must survive\n')
        backups=list(checkout.glob('sky-position-backup-*'))
        assert len(backups)==1
        assert (backups[0]/'gwak/train/dataloader.py').read_bytes()==initial
        assert (checkout/'gwak/sky_position.py').is_file()
    else:
        assert result.returncode!=0
        assert target.read_bytes()==initial
        assert not (checkout/'gwak/sky_position.py').exists()
        assert not list(checkout.glob('sky-position-backup-*'))
