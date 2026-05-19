"""Client/server integration shims.

Two thin override classes that you can drop into the existing wiring without
forking LeRobot. Once a real codec (e.g. MS-ILLM) is registered, set the codec
name via env / param and the rest of the pipeline keeps working unchanged.

Usage on the CLIENT (rosetta_client_node.py, around line 372):

    from rosetta.common.compression.integration import CompressedRobotClient
    client = CompressedRobotClient(config, codec_name="msillm")  # was RobotClient

Usage on the SERVER (custom entry point that wraps lerobot.async_inference.policy_server):

    from rosetta.common.compression.integration import wrap_policy_server
    wrap_policy_server()                # monkeypatches the receive path
    from lerobot.async_inference import policy_server
    policy_server.serve(...)

The server-side hook decompresses every CompressedImagePayload it sees in an
incoming observation right after pickle.loads, before the policy runs.
"""
from __future__ import annotations

import pickle
from typing import Any

from .base import get
from .observation import compress_observation, decompress_observation


# --- CLIENT SHIM ---------------------------------------------------------

def CompressedRobotClient(*args, codec_name: str = "identity", **kwargs):
    """Factory that returns a RobotClient subclass with image compression.

    Imported lazily so this module stays usable without LeRobot installed
    (e.g. for tests or when only the server side is being run).
    """
    from lerobot.async_inference.robot_client import RobotClient

    codec = get(codec_name)

    class _Client(RobotClient):
        def send_observation(self, obs):  # type: ignore[override]
            inner = getattr(obs, "observation", None)
            if isinstance(inner, dict):
                obs.observation = compress_observation(inner, codec)
            return super().send_observation(obs)

    return _Client(*args, **kwargs)


# --- SERVER SHIM ---------------------------------------------------------

def decompress_inplace(obs_obj: Any) -> Any:
    """Apply decompress_observation to whichever dict carries the images.

    LeRobot wraps the dict inside a TimedObservation; older paths pass the
    dict directly. Handle both.
    """
    inner = getattr(obs_obj, "observation", None)
    if isinstance(inner, dict):
        obs_obj.observation = decompress_observation(inner)
        return obs_obj
    if isinstance(obs_obj, dict):
        return decompress_observation(obs_obj)
    return obs_obj


_WRAPPED = False


def wrap_policy_server() -> None:
    """Install an image-decompression hook that fires after every pickle.loads.

    Patches ``pickle.loads`` on the cached ``pickle`` module object so all
    callers (the policy server, its dependencies, anything launched via
    ``runpy.run_module``) automatically apply ``decompress_inplace`` to the
    deserialized value. ``decompress_inplace`` is a no-op for any object that
    isn't a TimedObservation/dict containing CompressedImagePayload entries,
    so the patch is safe to leave on globally.

    Idempotent: a second call is a no-op.

    Why patch ``pickle.loads`` rather than ``policy_server.pickle.loads``:
    ``runpy.run_module(..., run_name='__main__')`` re-executes the policy
    server source in a fresh namespace, so its ``import pickle`` line rebinds
    ``pickle`` to whatever ``sys.modules['pickle']`` is at execution time. The
    only place a patch reliably survives that flow is on the shared
    ``pickle`` module object itself.
    """
    global _WRAPPED
    if _WRAPPED:
        return

    original_loads = pickle.loads

    def patched_loads(buf, *a, **kw):
        return decompress_inplace(original_loads(buf, *a, **kw))

    pickle.loads = patched_loads  # type: ignore[assignment]
    _WRAPPED = True
