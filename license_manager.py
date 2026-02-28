import os
import json
import requests

# bring in device handling
from device_manager import ensure_device_id

# URL of the running license server
# updated to production endpoint per user request
LICENSE_SERVER = 'https://api-licenses.cepac-eg.com/'

# local storage path for the license/trial key (sibling to this module)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LICENSE_FILE = os.path.join(BASE_DIR, 'license.json')


def load_license_key():
    """Return the stored key or None if not present or invalid."""
    if not os.path.exists(LICENSE_FILE):
        return None
    try:
        with open(LICENSE_FILE, 'r') as f:
            data = json.load(f)
            return data.get('key')
    except Exception:
        return None


def save_license_key(key: str):
    """Persist a license or trial key locally."""
    try:
        with open(LICENSE_FILE, 'w') as f:
            json.dump({'key': key}, f)
    except Exception:
        # best effort; calling code handles any errors via retries/flashes
        pass


def _get(path, params=None):
    url = LICENSE_SERVER.rstrip('/') + path
    return requests.get(url, params=params, timeout=5)


def _get_with_device(path, params=None):
    # ensure device id exists and attach to params
    device_id = ensure_device_id()
    params = params or {}
    params['device_id'] = device_id
    return _get(path, params)


def _post_with_device(path, json_payload=None):
    device_id = ensure_device_id()
    payload = json_payload or {}
    payload['device_id'] = device_id
    return _post(path, json_payload=payload)


def _post(path, json_payload=None):
    url = LICENSE_SERVER.rstrip('/') + path
    return requests.post(url, json=json_payload, timeout=5)


def verify_license(key: str):
    """Call the license server to verify a full license key.

    Device id is automatically appended.  See ``_get_with_device``.
    Returns the parsed JSON response (or ``{'valid': False}`` on
    non‑200 status), or ``None`` if the server cannot be reached.
    """
    try:
        resp = _get_with_device('/api/verify_license', params={'key': key})
        if resp.status_code == 200:
            return resp.json()
        else:
            return {'valid': False}
    except requests.RequestException:
        return None


def verify_trial(key: str):
    """Call the license server to verify a trial key."""
    try:
        resp = _get_with_device('/api/verify_trial', params={'key': key})
        if resp.status_code == 200:
            return resp.json()
        else:
            return {'valid': False}
    except requests.RequestException:
        return None


def create_trial(email: str):
    """Ask the license server to create a new trial for the given email.

    The device id is added automatically to the payload.
    Returns the parsed JSON on success, ``None`` on network failure or
    non‑200 status (a failure may also come back with ``{'success':False}``).
    """
    try:
        resp = _post_with_device('/api/create_trial', json_payload={'email': email})
        if resp.status_code == 200:
            return resp.json()
        else:
            # server replied error (eg 409 if device already has trial)
            try:
                return resp.json()
            except Exception:
                return None
    except requests.RequestException:
        return None


def check_trial_eligibility():
    """Ask the server whether the current device may start a trial.

    Returns the parsed JSON response, or ``None`` if the server is unreachable.
    """
    try:
        resp = _get_with_device('/api/check_trial')
        if resp.status_code == 200:
            return resp.json()
        else:
            return {'eligible': False}
    except requests.RequestException:
        return None


def check_stored_license():
    """Inspect the saved key and confirm with the license server.

    Device id will also be included automatically.
    Returns a tuple ``(status, info)`` where ``status`` is:

    * ``True`` if the saved key is valid,
    * ``False`` if the key was present but invalid/expired, and
    * ``None`` if the server could not be contacted.

    ``info`` carries either the response data or a short message.
    """
    key = load_license_key()
    if not key:
        return False, 'no key'

    if key.startswith('TRIAL-'):
        result = verify_trial(key)
    else:
        result = verify_license(key)

    if result is None:
        return None, 'server unreachable'
    if result.get('valid'):
        return True, result
    return False, result
