from anima_def_dtp.config import get_settings
from anima_def_dtp.constants import PAPER_ARXIV_ID


def test_settings_use_correct_paper_identity() -> None:
    settings = get_settings()
    assert settings.codename == "DEF-DTP"
    assert settings.paper_arxiv == PAPER_ARXIV_ID
    assert settings.project_name == "anima-def-dtp"
