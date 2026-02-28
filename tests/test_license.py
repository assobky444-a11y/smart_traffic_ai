import os
import sys
import json
import pytest
from flask import url_for

# make sure top-level package path is on sys.path for imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

from license_manager import (
    LICENSE_FILE,
    load_license_key,
    save_license_key,
    verify_license,
    verify_trial,
    create_trial,
    check_stored_license,
)

import app as app_module


class DummyResp:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


def test_license_file_roundtrip(tmp_path, monkeypatch):
    # ensure license module uses temp file instead of project root
    temp_file = str(tmp_path / 'license.json')
    monkeypatch.setattr('license_manager.LICENSE_FILE', temp_file)

    assert load_license_key() is None
    save_license_key('ABC')
    assert load_license_key() == 'ABC'
    # corrupt file returns None
    with open(temp_file, 'w') as f:
        f.write('not json')
    assert load_license_key() is None


def test_device_manager(tmp_path, monkeypatch):
    import device_manager
    # redirect device file
    monkeypatch.setattr('device_manager.DEVICE_FILE', str(tmp_path / 'device.json'))
    assert device_manager.load_device_id() is None
    did = device_manager.ensure_device_id()
    assert did.startswith('DEVICE-')
    # subsequent calls return same value
    assert device_manager.ensure_device_id() == did
    assert device_manager.load_device_id() == did

    # remove file and ensure regenerated id matches (hardware id scenario)
    if os.path.exists(device_manager.DEVICE_FILE):
        os.remove(device_manager.DEVICE_FILE)
    did2 = device_manager.ensure_device_id()
    assert did2 == did  # deterministic MAC-based id keeps constant


def test_verify_functions(monkeypatch):
    # monkeypatch device id
    monkeypatch.setattr('license_manager.ensure_device_id', lambda: 'DEVICE-XYZ')

    # intercept underlying _get to verify device_id parameter is passed
    captured = {}
    def fake_get(path, params, timeout):
        captured['path'] = path
        captured['params'] = params
        return DummyResp(200, {'valid': True})
    monkeypatch.setattr('license_manager', '_get', fake_get)

    assert verify_license('KEY')['valid'] is True
    assert captured['params']['device_id'] == 'DEVICE-XYZ'
    captured.clear()
    assert verify_trial('KEY')['valid'] is True
    assert captured['params']['device_id'] == 'DEVICE-XYZ'

    # network error should return None
    monkeypatch.setattr('license_manager', '_get', lambda path, params, timeout: (_ for _ in ()).throw(Exception('oops')))
    assert verify_license('KEY') is None
    assert verify_trial('KEY') is None

    # create_trial should send device_id in POST
    captured_post = {}
    def fake_post(path, json, timeout):
        captured_post['path'] = path
        captured_post['json'] = json
        return DummyResp(200, {'trial_key': 'TRIAL-1'})
    monkeypatch.setattr('license_manager', '_post', fake_post)
    assert create_trial('x@x.com')['trial_key'] == 'TRIAL-1'
    assert captured_post['json']['device_id'] == 'DEVICE-XYZ'
    monkeypatch.setattr('license_manager', '_post', lambda path, json, timeout: (_ for _ in ()).throw(Exception('oops')))
    assert create_trial('x@x.com') is None


def test_check_stored_license(tmp_path, monkeypatch):
    # use temporary license file
    monkeypatch.setattr('license_manager.LICENSE_FILE', str(tmp_path / 'license.json'))
    # no key
    assert check_stored_license() == (False, 'no key')

    # valid full license
    save_license_key('LICENSE-1')
    monkeypatch.setattr('license_manager.verify_license', lambda k: {'valid': True})
    assert check_stored_license()[0] is True

    # invalid full license
    monkeypatch.setattr('license_manager.verify_license', lambda k: {'valid': False})
    assert check_stored_license()[0] is False

    # trial key
    save_license_key('TRIAL-ABC')
    monkeypatch.setattr('license_manager.verify_trial', lambda k: {'valid': True})
    assert check_stored_license()[0] is True

    # unreachable
    monkeypatch.setattr('license_manager.verify_trial', lambda k: None)
    assert check_stored_license()[0] is None


def test_activation_flow(monkeypatch, tmp_path):
    # redirect both license and device files to temp directory
    monkeypatch.setattr('license_manager.LICENSE_FILE', str(tmp_path / 'license.json'))
    import device_manager
    monkeypatch.setattr('device_manager.DEVICE_FILE', str(tmp_path / 'device.json'))
    # ensure_device_id will create a stable id in tmp path
    device_id = device_manager.ensure_device_id()
    monkeypatch.setattr('license_manager.ensure_device_id', lambda: device_id)
    client = app_module.app.test_client()

    # license server not reachable: ensure redirect to license page
    monkeypatch.setattr('license_manager.check_stored_license', lambda: (None, 'unreachable'))
    res = client.get('/login', follow_redirects=True)
    assert b'License Activation' in res.data

    # UI visibility checks at /license
    # no file -> free trial section present
    # ensure file does not exist
    if os.path.exists(LICENSE_FILE):
        os.remove(LICENSE_FILE)
    # server indicates eligibility
    monkeypatch.setattr('license_manager.check_trial_eligibility', lambda: {'eligible': True})
    res = client.get('/license')
    assert b'Start Free Trial' in res.data
    assert b'Device ID:' in res.data

    # if server says not eligible, the form is hidden and reason shown
    monkeypatch.setattr('license_manager.check_trial_eligibility', lambda: {'eligible': False, 'reason':'already used'})
    res = client.get('/license')
    assert b'Start Free Trial' not in res.data
    assert b'already used' in res.data

    # stored active trial -> hide free-trial forms
    with open(LICENSE_FILE, 'w') as f:
        json.dump({'key': 'TRIAL-ABC'}, f)
    monkeypatch.setattr('license_manager.verify_trial', lambda k: {'valid': True, 'expiry_date':'2099-01-01'})
    res = client.get('/license')
    assert b'Start Free Trial' not in res.data
    assert b'Enter License Key' in res.data

    # stored expired trial -> hide trial, show expiry message and keep file
    monkeypatch.setattr('license_manager.verify_trial', lambda k: {'valid': False})
    res = client.get('/license')
    assert b'Start Free Trial' not in res.data
    assert b'Trial expired' in res.data
    # ensure file still present so user can't obtain a second trial
    assert os.path.exists(LICENSE_FILE)

    # attempt to trigger ensure_license redirect path, which previously would
    # remove the file.  simulate invalid check returning False
    monkeypatch.setattr('license_manager.check_stored_license', lambda: (False, {'valid': False}))
    res = client.get('/somepage', follow_redirects=True)
    # after redirect user still not allowed to get a new trial (file stays)
    assert os.path.exists(LICENSE_FILE)

    # stored full license -> hide trial section
    with open(LICENSE_FILE, 'w') as f:
        json.dump({'key': 'LICENSE-123'}, f)
    res = client.get('/license')
    assert b'Start Free Trial' not in res.data

    # simulate successful activation via form
    monkeypatch.setattr('license_manager.verify_license', lambda k: {'valid': True})
    res = client.post('/license/activate', data={'license_key': 'KEY'}, follow_redirects=True)
    assert b'License activated successfully' in res.data
    # key should be saved
    assert json.load(open(LICENSE_FILE))['key'] == 'KEY'

    # simulate license activation failing with custom server message
    monkeypatch.setattr('license_manager.verify_license', lambda k: {'valid': False, 'reason': 'Used on other device'})
    res = client.post('/license/activate', data={'license_key': 'BAD'}, follow_redirects=True)
    assert b'Used on other device' in res.data

    # simulate trial creation
    def fake_create(email):
        return {'success': True, 'trial_key': 'TRIAL-123'}
    monkeypatch.setattr('license_manager.create_trial', fake_create)
    monkeypatch.setattr('license_manager.verify_trial', lambda k: {'valid': True})
    res = client.post('/license/start_trial', data={'email': 'a@b.com'}, follow_redirects=True)
    assert b'Free trial started' in res.data
    assert json.load(open(LICENSE_FILE))['key'].startswith('TRIAL-')

    # simulate server rejecting trial because device already used
    monkeypatch.setattr('license_manager.create_trial', lambda e: {'success': False, 'reason': 'Device already used'})
    if os.path.exists(LICENSE_FILE):
        os.remove(LICENSE_FILE)
    res = client.post('/license/start_trial', data={'email': 'a@b.com'}, follow_redirects=True)
    assert b'Device already used' in res.data
    assert not os.path.exists(LICENSE_FILE)

    # verifying an existing trial key
    monkeypatch.setattr('license_manager.verify_trial', lambda k: {'valid': True, 'expiry_date': '2026-12-31'})
    res = client.post('/license/verify_trial', data={'trial_key': 'TRIAL-XYZ'}, follow_redirects=True)
    assert b'Trial Active' in res.data

