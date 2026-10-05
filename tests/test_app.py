import os
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('RUN_INTEGRATION') != '1', reason='Requires downloaded real model and corpus')
def test_real_streamlit_search_and_bad_input():
    app = AppTest.from_file(__import__('pathlib').Path('streamlit_app.py').resolve(), default_timeout=120).run()
    assert not app.exception
    app.text_input[0].set_value('Vitamin D deficiency causes rickets.')
    app.button[0].click().run()
    assert not app.exception
    assert len(app.expander) == 5
    assert any('BEIR SciFact corpus ID:' in c.value for c in app.caption)
    app.text_input[0].set_value('   ')
    app.button[0].click().run()
    assert len(app.warning) == 1
    assert not app.exception
