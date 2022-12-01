from af22c.utils import get_raw_proteome_name


def test_proteome_name_extraction_tar_gz():
    proteome_name = "FOO0190_ALPHA_42"
    filename = f"/home/foo/proteins/{proteome_name}.tar.gz"
    assert get_raw_proteome_name(filename) == proteome_name


def test_proteome_name_extraction_dir():
    proteome_name = "FOO0190_ALPHA_42"
    filename = f"/home/foo/proteins/{proteome_name}"
    assert get_raw_proteome_name(filename) == proteome_name
