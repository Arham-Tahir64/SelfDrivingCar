from pathlib import Path

from autonomy_demo.interfaces.types import ReplayFrame
from autonomy_demo.replay.reader import JsonReplayReader
from autonomy_demo.replay.writer import Hdf5OrJsonReplayWriter


def test_replay_round_trip(tmp_path: Path) -> None:
    writer = Hdf5OrJsonReplayWriter(tmp_path)
    writer.record(
        ReplayFrame(
            tick_id=0,
            sim_time_s=0.0,
            topic_payloads={"control/vehicle_command": {"throttle": 0.1}},
        )
    )
    path = writer.finalize()
    frames = JsonReplayReader(path).read()
    assert len(frames) == 1
    assert frames[0].tick_id == 0

