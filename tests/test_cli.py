import argparse
from datetime import time

import pytest

from outfit.cli import scale_range, trip_event
from outfit.schemas import Occasion
from outfit.tools.weather import Transport


def test_scale_range():
    assert scale_range("3") == (3, 3) and scale_range("2-4") == (2, 4)
    with pytest.raises(argparse.ArgumentTypeError):
        scale_range("4-6")


def test_trip_event_reads_day_occasion_time_hours_and_transport():
    event = trip_event("2, party, 20:00, 4, 0.5, car")

    assert (event.day, event.occasion, event.start) == (2, Occasion.PARTY, time(20, 0))
    assert (event.hours_away, event.hours_outdoors, event.transport) == (4, 0.5, Transport.CAR)


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("2,party,20:00", "expected DAY,OCCASION"),
        ("2,disco,20:00,4,0.5,car", "occasion"),
        ("2,party,20:00,4,5,car", "hours_outdoors cannot exceed hours_away"),
    ],
)
def test_trip_event_says_what_is_wrong(text, says):
    with pytest.raises(argparse.ArgumentTypeError, match=says):
        trip_event(text)
