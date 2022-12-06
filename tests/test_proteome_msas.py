import os
import textwrap
import tempfile
import tarfile
from contextlib import contextmanager
from io import BytesIO
from af22c.proteome import ProteomeMSAs

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
def archived_sample_msa(compression=None):
    """
    Create a temporary archive that contains the files specified in the `MSA_FIXTURES` constant.
    """
    with tempfile.TemporaryDirectory() as temp_dirname:
        mode = "w"
        ext = ".tar"
        if compression:
            mode += ":" + compression
            ext += "." + compression
        temp_tar_path = os.path.join(temp_dirname, "alien" + ext)
        with tarfile.open(temp_tar_path, mode=mode) as temp_tar:
            # add predefined fixture contents
            for path, content in MSA_FIXTURES.items():
                add_text_as_file(temp_tar, path, content)

        yield temp_tar_path


def test_load_sample_proteome_from_archive():
    with archived_sample_msa() as archive_path:
        msas = ProteomeMSAs.from_archive(archive_path)
        assert msas.get_uniprot_ids() == VALID_PROTEIN_IDS
