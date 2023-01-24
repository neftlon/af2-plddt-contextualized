import os
import textwrap
import tempfile
import tarfile
from contextlib import contextmanager
from io import BytesIO

from af22c.load_msa import MsaMatchAttribs
from af22c.proteome import ProteomeMSAs
import pytest

MSA_FIXTURES = {
    "alien/msas/F0042R.a3m": textwrap.dedent("""\
    >F0042R
    SEQWENCE"""),
    "alien/msas/B4R43L.a3m": textwrap.dedent("""\
    >B4R43L
    QWINCY
    >B1111I
    -W-NCY
    >B2222Z
    --IN--"""),
    "alien/N0TF0UND.a3m": textwrap.dedent("""\
    >since this file is not in the msas/ subdir, this file should not be found by default
    QWEEP
    """)
}
VALID_PROTEIN_IDS = {"F0042R", "B4R43L"}

# NB: the scores in the following MSA headers are just hallucinated and are not reflected by the MSA itself
MSA_WITH_PARAMS_FIXTURE = {
    # hallucinated file content
    "content": {
        "alien/msas/111.a3m": textwrap.dedent("""\
        >111
        SEQ
        >112\t42\t0.66\t0\t0\t3\t3\t0\t3\t3
        EEQ
        >113\t12\t0.33\t0\t1\t2\t2\t1\t2\t2
        -E-
        """),
    },
    "match_attribs": [
        MsaMatchAttribs(42, 0.66, 0, 0, 3, 3, 0, 3, 3),
        MsaMatchAttribs(12, 0.33, 0, 1, 2, 2, 1, 2, 2),
    ]
}


def add_directory(tar: tarfile.TarFile, name: str):
    """
    Add a directory with `name` to the `tar` file.
    """
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    tar.addfile(info)


def add_text_as_file(tar: tarfile.TarFile, path: str, text: str):
    """
    Add a file at `path` to the `tar` file containing `text`.
    """
    data = text.encode("utf-8")
    buf = BytesIO(data)
    info = tarfile.TarInfo(name=path)
    info.size = len(data)
    tar.addfile(tarinfo=info, fileobj=buf)


@contextmanager
def archived_sample_msa(compression=None, fixture=None):
    """
    Create a temporary archive that contains the files specified in the `MSA_FIXTURES` constant.
    """
    if fixture is None:
        fixture = MSA_FIXTURES

    with tempfile.TemporaryDirectory() as temp_dirname:
        mode = "w"
        ext = ".tar"
        if compression:
            mode += ":" + compression
            ext += "." + compression
        temp_tar_path = os.path.join(temp_dirname, "alien" + ext)
        with tarfile.open(temp_tar_path, mode=mode) as temp_tar:
            # add predefined fixture contents
            for path, content in fixture.items():
                add_text_as_file(temp_tar, path, content)

        yield temp_tar_path


@pytest.mark.parametrize("compression", [None, "gz"])
def test_load_sample_proteome_from_archive(compression):
    with archived_sample_msa(compression=compression) as archive_path:
        # check that MSAs can be loaded at all
        msas = ProteomeMSAs.from_archive(archive_path)
        assert msas.get_uniprot_ids() == VALID_PROTEIN_IDS

        # check that protein gathering by index is possible
        for prot_id in VALID_PROTEIN_IDS:
            msa = msas[prot_id]
            assert msa is not None

        # check that all MSAs can be extracted at once
        for msa in msas.get_msas():
            assert msa is not None
            assert msa.query_id in VALID_PROTEIN_IDS


def test_sample_msa():
    with archived_sample_msa(fixture=MSA_WITH_PARAMS_FIXTURE["content"]) as archive_path:
        msas = ProteomeMSAs.from_archive(archive_path)
        msas = list(msas.get_msas())
        assert len(msas) == 1
        msa = msas[0]
        for match, expected_attribs in zip(msa.matches, MSA_WITH_PARAMS_FIXTURE["match_attribs"]):
            assert match.attribs == expected_attribs
