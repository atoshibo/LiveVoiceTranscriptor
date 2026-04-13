"""
UI Smoke Tests - Minimal backend/UI route health checks.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestUISmoke:
    """Minimal UI and route health checks."""

    def test_ui_html_not_empty(self):
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert len(html) > 1000
        assert "LiveVoiceTranscriptor" in html

    def test_ui_has_auth_gate(self):
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "auth-gate" in html
        assert "token" in html.lower()

    def test_ui_has_session_views(self):
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "page-sessions" in html
        assert "page-session-detail" in html

    def test_ui_has_upload_page(self):
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "page-upload" in html
        assert "upload-file" in html

    def test_ui_has_model_view(self):
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "page-models" in html

    def test_ui_has_diagnostics(self):
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "page-diagnostics" in html

    def test_ui_uses_localstorage_not_url(self):
        """Token must not be in URL query strings."""
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "localStorage" in html
        # Should use Bearer header
        assert "Bearer" in html

    def test_ui_has_srt_vtt_links(self):
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "subtitle.srt" in html
        assert "subtitle.vtt" in html

    def test_ui_has_pipeline_debug(self):
        """Must have visible pipeline/provenance debug info."""
        from app.ui.dashboard import get_ui_html
        html = get_ui_html()
        assert "pipeline" in html.lower()

    def test_fastapi_app_importable(self):
        from app.main import app
        assert app.title == "LiveVoiceTranscriptor"

    def test_api_v2_router_mounted(self):
        from app.main import app
        routes = [r.path for r in app.routes]
        assert "/api/v2/health" in routes or any("/api/v2" in str(r) for r in app.routes)
