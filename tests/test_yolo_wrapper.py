import pytest

from tdb.models.yolo import _check_names


def test_matching_names_pass():
    _check_names({0: "car", 1: "bus", 2: "truck"}, ["car", "bus", "truck"])
    _check_names(["car", "bus"], ["car", "bus"])


def test_wrong_variant_is_rejected():
    with pytest.raises(ValueError, match="wrong dataset variant"):
        _check_names({0: "car", 1: "bus", 2: "truck"}, ["big bus", "big truck"])
