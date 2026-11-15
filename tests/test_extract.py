"""Place and date extraction from chat messages and bare CLI places."""

from datetime import date

from hike_agent.extract import extract_hike_request

SATURDAY = date(2026, 11, 14)
SUNDAY = date(2026, 11, 15)
FRIDAY = date(2026, 11, 13)


def test_bare_place_is_left_alone():
    found = extract_hike_request("Boulder, Colorado", today=SATURDAY)
    assert found["place"] == "Boulder, Colorado"
    assert found["when"] == {}


def test_me_stays_a_place():
    found = extract_hike_request("me", today=SATURDAY)
    assert found["place"] == "me"
    assert found["when"] == {}


def test_near_place_and_this_weekend_on_saturday():
    found = extract_hike_request("hikes near Boulder this weekend", today=SATURDAY)
    assert found["place"] == "Boulder"
    assert found["when"]["start"] == "2026-11-14"
    assert found["when"]["end"] == "2026-11-15"
    assert "this weekend" in found["when"]["label"]


def test_city_and_state_survive_the_weekend_phrase():
    found = extract_hike_request("hikes near Boulder, Colorado this weekend", today=FRIDAY)
    assert found["place"] == "Boulder, Colorado"
    assert found["when"]["start"] == "2026-11-14"
    assert found["when"]["end"] == "2026-11-15"


def test_this_weekend_on_sunday_is_the_current_weekend():
    found = extract_hike_request("near Estes Park this weekend", today=SUNDAY)
    assert found["place"] == "Estes Park"
    assert found["when"]["start"] == "2026-11-14"
    assert found["when"]["end"] == "2026-11-15"


def test_next_weekend_is_the_following_saturday():
    found = extract_hike_request("hikes around Moab next weekend", today=SATURDAY)
    assert found["place"] == "Moab"
    assert found["when"]["start"] == "2026-11-21"
    assert found["when"]["end"] == "2026-11-22"


def test_named_date_and_in_phrase():
    found = extract_hike_request("hikes in Rocky Mountain National Park on November 20", today=SATURDAY)
    assert found["place"] == "Rocky Mountain National Park"
    assert found["when"]["start"] == "2026-11-20"
    assert found["when"]["end"] == "2026-11-20"


def test_tomorrow_moves_the_forecast_day():
    found = extract_hike_request("tomorrow in Yosemite Valley", today=SATURDAY)
    assert found["place"] == "Yosemite Valley"
    assert found["when"]["start"] == "2026-11-15"


def test_near_wins_over_an_earlier_in():
    found = extract_hike_request("hiking in the rain near Boulder this weekend", today=SATURDAY)
    assert found["place"] == "Boulder"
