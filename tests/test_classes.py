import pytest

from tdb.data.classes import ClassConfig, remap_label_lines


@pytest.fixture
def cfg():
    return ClassConfig.load()


def test_config_covers_all_original_classes(cfg):
    assert len(cfg.original) == 12
    assert set(cfg.id_map()) == set(range(12))


def test_known_mappings(cfg):
    m = cfg.id_map()
    car, bus, truck = (cfg.merged.index(n) for n in ("car", "bus", "truck"))
    assert m[cfg.original.index("car")] == car
    assert m[cfg.original.index("bus-l-")] == bus
    assert m[cfg.original.index("truck-xl-")] == truck


def test_remap_keeps_coordinates_and_drops_blank_lines(cfg):
    lines = ["4 0.5 0.5 0.1 0.2", "", "11 0.1 0.2 0.3 0.4\n"]
    out = remap_label_lines(lines, cfg.id_map())
    assert out == ["0 0.5 0.5 0.1 0.2", "2 0.1 0.2 0.3 0.4"]


def test_remap_rejects_unknown_class(cfg):
    with pytest.raises(ValueError):
        remap_label_lines(["12 0.5 0.5 0.1 0.1"], cfg.id_map())


def test_validate_catches_missing_mapping():
    bad = ClassConfig(original=["car", "van"], merged=["car"], mapping={"car": "car"}, pcu={})
    with pytest.raises(ValueError):
        bad.validate()
