import os
import json
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_FILE = os.path.join(BASE_DIR, 'device.json')


def load_device_id():
    """Return stored device ID or None if not present or unreadable."""
    if not os.path.exists(DEVICE_FILE):
        return None
    try:
        with open(DEVICE_FILE, 'r') as f:
            data = json.load(f)
            return data.get('device_id')
    except Exception:
        return None


def generate_device_id():
    """Create a new unique device identifier."""
    return 'DEVICE-' + uuid.uuid4().hex.upper()


def _compute_hardware_id():
    """Derive a stable device identifier from machine hardware.

    Currently we use the MAC address returned by uuid.getnode(). If the MAC
    looks like a real hardware address (not a randomly generated one) we
    hash it to produce a short stable ID.  If no usable MAC is available we
    fall back to ``None`` which triggers random generation.
    """
    try:
        mac = uuid.getnode()
        # According to documentation, if the 8th bit is set the MAC is random
        if (mac >> 40) & 0x01:
            return None
        # canonical hex string
        mac_str = f"{mac:012x}"
        import hashlib
        h = hashlib.sha256(mac_str.encode()).hexdigest().upper()
        return 'DEVICE-' + h[:16]
    except Exception:
        return None


def ensure_device_id():
    """Ensure a device ID exists and return it.

    If ``device.json`` is missing we attempt to compute a hardware‑based ID
    so that deleting the file does **not** change the identifier.  The value
    (computed or newly generated) is always written back to the file for
    fast lookup.
    """
    existing = load_device_id()
    if existing:
        return existing
    # try deterministic hardware ID first
    hw = _compute_hardware_id()
    new_id = hw or generate_device_id()
    try:
        with open(DEVICE_FILE, 'w') as f:
            json.dump({'device_id': new_id}, f)
    except Exception:
        pass
    return new_id
