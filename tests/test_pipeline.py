"""
test_pipeline_smoke.py
======================
Smoke tests for signal_kernel/pipeline.py — NO mocks. Real everything.

Uses the real registry, real validator, real hardcoded .sh recipes, and
real subprocess execution. Hardcoded recipes (NEWS_ARTICLE, SAAS_DOCS,
REST_API_JSON, JSON_LD_STRUCTURED, ECOMMERCE_PRODUCT) skip validator
dry-run by design, so no test fixtures directory is required.

Run:
    pytest test_pipeline_smoke.py -v
"""

from __future__ import annotations

import uuid # noqa
from typing import Optional

import pytest

from signal_kernel.contracts import (
    KernelInput,
    KernelOutput,
    TopologyClassStr,
    new_run_id,
)
from signal_kernel.pipeline import (
    PipelineConfig,
    execute_sync,
    initialize,
    is_initialized,
    reset,
    shutdown_sync,
    _DEFAULT_CONFIG, # noqa
    _warm, # noqa
)
from signal_kernel.recipes import registry, validator


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_SHA256 = "a" * 64  # placeholder intent_vector_hash (never validated for content)


def _ki(
    raw_content: str,
    topology_class: str = "SAAS_DOCS",
    content_type: str = "html",
    source_url: str = "https://docs.stripe.com/api",
    run_id: Optional[str] = None,
) -> KernelInput:
    return KernelInput(
        raw_content=raw_content,
        topology_class=TopologyClassStr(topology_class),
        intent_vector_hash=_SHA256,          # type: ignore[arg-type]
        content_type=content_type,           # type: ignore[arg-type]
        source_url=source_url,               # type: ignore[arg-type]
        run_id=run_id or new_run_id(),
    )


# Minimal HTML that saas_docs.sh will extract from.
# The recipe's awk state machine fires on <main> ... </main>.
SAAS_HTML_WITH_MAIN = """\
<!DOCTYPE html>
<html>
<head><title>Stripe API Reference</title></head>
<body>
  <nav>Site navigation noise that should be stripped</nav>
  <main>
    <h1>Authentication</h1>
    <p>The Stripe API uses API keys to authenticate requests.</p>
    <p>Pass your API key in the Authorization header like this:</p>
    <pre><code>Authorization: Bearer sk_test_YOUR_KEY</code></pre>
    <h2>Errors</h2>
    <p>Stripe uses conventional HTTP response codes to indicate success or failure.</p>
  </main>
  <footer>Footer noise that should be stripped</footer>
</body>
</html>
"""

# HTML with no <main> tag — saas_docs.sh awk machine never enters signal zone,
# so stdout is empty -> extraction_empty=True.
SAAS_HTML_NO_MAIN = """\
<!DOCTYPE html>
<html>
<head><title>No Main Tag</title></head>
<body>
  <div class="content">
    <p>This content is not inside a main tag.</p>
  </div>
</body>
</html>
"""

JSON_LD_CONTENT = """\
<html>
<head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Article","headline":"Test headline",
"description":"Test description of the article content here.","author":{"@type":"Person","name":"Jane Doe"}}
</script>
</head>
<body><p>Body text</p></body>
</html>
"""

ECOMMERCE_HTML = """\
<!DOCTYPE html>
<html>
<body>
  <div class="product">
    <h1 class="product-title">Blue Widget Pro</h1>
    <span class="price">$29.99</span>
    <div class="product-description">
      <p>A high-quality widget suitable for all your widgeting needs.</p>
    </div>
  </div>
</body>
</html>
"""

REST_JSON = """\
{
  "results": [
    {"id": 1, "name": "Alice", "role": "admin"},
    {"id": 2, "name": "Bob",   "role": "user"}
  ],
  "total": 2
}
"""


@pytest.fixture(autouse=True)
def clean_pipeline():
    """Isolate pipeline module state between every test."""
    reset()
    yield
    reset()


# =============================================================================
# 1. MODULE-LEVEL SANITY
# =============================================================================

class TestModuleSanity:
    def test_default_config_is_valid(self):
        assert isinstance(_DEFAULT_CONFIG, PipelineConfig)
        assert _DEFAULT_CONFIG.subprocess_timeout_ms >= 100
        assert _DEFAULT_CONFIG.spawn_timeout_ms >= 100
        assert 1 <= _DEFAULT_CONFIG.max_spawn_attempts <= 5

    def test_not_initialized_before_initialize(self):
        assert not is_initialized()

    def test_initialize_sets_flag(self):
        initialize()
        assert is_initialized()

    def test_initialize_idempotent(self):
        initialize()
        initialize()
        assert is_initialized()

    def test_reset_clears_flag(self):
        initialize()
        reset()
        assert not is_initialized()

    def test_shutdown_sync_safe_when_cold(self):
        shutdown_sync()  # must not raise


# =============================================================================
# 2. REAL REGISTRY RESOLVES HARDCODED RECIPES
# =============================================================================

class TestRealRegistry:
    def test_get_recipe_returns_recipe_mount_for_news_article(self):
        mount = registry.get_recipe("NEWS_ARTICLE")
        assert mount.topology_class == "NEWS_ARTICLE"
        assert mount.is_hardcoded is True

    def test_get_recipe_returns_recipe_mount_for_saas_docs(self):
        mount = registry.get_recipe("SAAS_DOCS")
        assert mount.topology_class == "SAAS_DOCS"
        assert mount.is_hardcoded is True

    def test_get_recipe_returns_recipe_mount_for_rest_api_json(self):
        mount = registry.get_recipe("REST_API_JSON")
        assert mount.topology_class == "REST_API_JSON"
        assert mount.is_hardcoded is True

    def test_get_recipe_returns_recipe_mount_for_json_ld_structured(self):
        mount = registry.get_recipe("JSON_LD_STRUCTURED")
        assert mount.topology_class == "JSON_LD_STRUCTURED"
        assert mount.is_hardcoded is True

    def test_get_recipe_returns_recipe_mount_for_ecommerce_product(self):
        mount = registry.get_recipe("ECOMMERCE_PRODUCT")
        assert mount.topology_class == "ECOMMERCE_PRODUCT"
        assert mount.is_hardcoded is True

    def test_unknown_topology_falls_back_to_generic_html(self):
        mount = registry.get_recipe("TOTALLY_UNKNOWN_CLASS_XYZ")
        assert mount.topology_class == "GENERIC_HTML"

    def test_paywalled_falls_back_to_news_article(self):
        # PARENT_CLASS_MAP: NEWS_ARTICLE_PAYWALLED -> NEWS_ARTICLE
        mount = registry.get_recipe("NEWS_ARTICLE_PAYWALLED")
        assert mount.topology_class == "NEWS_ARTICLE"

    def test_recipe_hash_is_64_hex_chars(self):
        mount = registry.get_recipe("NEWS_ARTICLE")
        assert len(mount.recipe_hash) == 64
        assert all(c in "0123456789abcdef" for c in mount.recipe_hash)

    def test_recipe_path_points_to_existing_file(self):
        from pathlib import Path
        mount = registry.get_recipe("NEWS_ARTICLE")
        assert Path(mount.recipe_path).is_file()


# =============================================================================
# 3. REAL VALIDATOR PASSES HARDCODED RECIPES IMMEDIATELY
# =============================================================================

class TestRealValidator:
    def test_validator_passes_news_article(self):
        mount = registry.get_recipe("NEWS_ARTICLE")
        validator.check(mount)  # must not raise

    def test_validator_passes_saas_docs(self):
        mount = registry.get_recipe("SAAS_DOCS")
        validator.check(mount)

    def test_validator_passes_rest_api_json(self):
        mount = registry.get_recipe("REST_API_JSON")
        validator.check(mount)

    def test_validator_passes_json_ld_structured(self):
        mount = registry.get_recipe("JSON_LD_STRUCTURED")
        validator.check(mount)

    def test_validator_passes_ecommerce_product(self):
        mount = registry.get_recipe("ECOMMERCE_PRODUCT")
        validator.check(mount)


# =============================================================================
# 4. HAPPY PATH — REAL SUBPROCESS, REAL EXTRACTION
# =============================================================================

class TestHappyPath:
    def test_saas_docs_extracts_main_content(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert isinstance(result, KernelOutput)
        assert not result.extraction_empty
        assert result.run_id == ki.run_id
        assert result.topology_class == "SAAS_DOCS"
        assert result.clean_byte_count > 0
        assert result.latency_ms > 0

    def test_saas_docs_signal_contains_heading(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert "Authentication" in result.clean_signal

    def test_saas_docs_signal_strips_nav_noise(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert "navigation noise" not in result.clean_signal

    def test_saas_docs_signal_strips_footer_noise(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert "Footer noise" not in result.clean_signal

    def test_saas_docs_clean_bytes_less_than_raw(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.clean_byte_count < result.raw_byte_count

    def test_saas_docs_raw_byte_count_matches_input(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.raw_byte_count == ki.raw_byte_count

    def test_saas_docs_recipe_used_is_not_empty(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.recipe_used != ""

    def test_rest_api_json_returns_output(self):
        ki = _ki(
            REST_JSON,
            topology_class="REST_API_JSON",
            content_type="json",
            source_url="https://api.example.com/users",
        )
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert isinstance(result, KernelOutput)
        assert result.run_id == ki.run_id

    def test_json_ld_structured_returns_output(self):
        ki = _ki(JSON_LD_CONTENT, topology_class="JSON_LD_STRUCTURED")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert isinstance(result, KernelOutput)
        assert result.run_id == ki.run_id

    def test_ecommerce_product_returns_output(self):
        ki = _ki(ECOMMERCE_HTML, topology_class="ECOMMERCE_PRODUCT")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert isinstance(result, KernelOutput)
        assert result.run_id == ki.run_id

    def test_each_run_gets_unique_run_id(self):
        ki1 = _ki(SAAS_HTML_WITH_MAIN)
        ki2 = _ki(SAAS_HTML_WITH_MAIN)
        r1 = execute_sync(ki1, registry=registry, validator_check=validator.check)
        r2 = execute_sync(ki2, registry=registry, validator_check=validator.check)

        assert r1.run_id != r2.run_id

    def test_custom_config_accepted(self):
        cfg = PipelineConfig(subprocess_timeout_ms=8000)
        ki  = _ki(SAAS_HTML_WITH_MAIN)
        result = execute_sync(ki, registry=registry, validator_check=validator.check, config=cfg)

        assert isinstance(result, KernelOutput)


# =============================================================================
# 5. GRACEFUL DEGRADATION — SUBPROCESS PRODUCES NO OUTPUT
# =============================================================================

class TestGracefulDegradation:
    def test_html_with_no_main_tag_returns_empty_extraction(self):
        """
        news_article.sh only emits lines inside <article>...</article>.
        HTML without that tag -> awk never enters signal zone -> empty stdout.
        pipeline.py must degrade, not raise.
        """
        ki = _ki(SAAS_HTML_NO_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert isinstance(result, KernelOutput)
        assert result.extraction_empty is True
        assert result.clean_signal == ""

    def test_empty_extraction_preserves_run_id(self):
        ki = _ki(SAAS_HTML_NO_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.run_id == ki.run_id

    def test_empty_extraction_preserves_topology_class(self):
        ki = _ki(SAAS_HTML_NO_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.topology_class == "SAAS_DOCS"

    def test_empty_extraction_has_positive_latency(self):
        ki = _ki(SAAS_HTML_NO_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.latency_ms > 0

    def test_empty_extraction_raw_byte_count_still_set(self):
        ki = _ki(SAAS_HTML_NO_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.raw_byte_count == ki.raw_byte_count


# =============================================================================
# 6. UNKNOWN TOPOLOGY -> GENERIC_HTML FALLBACK
# =============================================================================

class TestGenericHtmlFallback:
    def test_unknown_topology_does_not_raise(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="COMPLETELY_UNKNOWN_CLASS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert isinstance(result, KernelOutput)

    def test_unknown_topology_result_has_correct_topology_class(self):
        """
        KernelOutput.topology_class reflects what the caller passed in,
        not the internal fallback recipe that was used.
        """
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="COMPLETELY_UNKNOWN_CLASS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.topology_class == "COMPLETELY_UNKNOWN_CLASS"


# =============================================================================
# 7. TOKEN REDUCTION AND SIGNAL DENSITY
# =============================================================================

class TestQualityMetrics:
    def test_token_reduction_pct_in_range(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert 0.0 <= result.token_reduction_pct <= 1.0

    def test_signal_density_between_zero_and_one(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert 0.0 <= result.signal_density <= 1.0

    def test_clean_bytes_never_exceed_raw_bytes(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.clean_byte_count <= result.raw_byte_count

    def test_token_delta_estimate_nonnegative(self):
        ki = _ki(SAAS_HTML_WITH_MAIN, topology_class="SAAS_DOCS")
        result = execute_sync(ki, registry=registry, validator_check=validator.check)

        assert result.token_delta_estimate >= 0


# =============================================================================
# 8. SEQUENTIAL CALLS — NO STATE BLEED
# =============================================================================

class TestSequentialCalls:
    def test_three_sequential_extractions_all_succeed(self):
        for i in range(3):
            ki = _ki(
                SAAS_HTML_WITH_MAIN.replace("Authentication", f"Section {i}"),
                run_id=new_run_id(),
            )
            result = execute_sync(ki, registry=registry, validator_check=validator.check)
            assert not result.extraction_empty
            assert result.run_id == ki.run_id

    def test_run_ids_unique_across_five_calls(self):
        ids = set()
        for _ in range(5):
            ki = _ki(SAAS_HTML_WITH_MAIN, run_id=new_run_id())
            result = execute_sync(ki, registry=registry, validator_check=validator.check)
            ids.add(result.run_id)

        assert len(ids) == 5

    def test_interleaved_topology_classes(self):
        """Switching topology class between calls must not corrupt output."""
        pairs = [
            ("SAAS_DOCS",          SAAS_HTML_WITH_MAIN, "html", "https://docs.stripe.com/api"),
            ("REST_API_JSON",       REST_JSON,              "json", "https://api.example.com/v1"),
            ("JSON_LD_STRUCTURED",  JSON_LD_CONTENT,        "html", "https://example.com/ld"),
        ]
        for tc, content, ct, url in pairs:
            ki = _ki(content, topology_class=tc, content_type=ct, source_url=url)
            result = execute_sync(ki, registry=registry, validator_check=validator.check)
            assert result.topology_class == tc
            assert result.run_id == ki.run_id


# =============================================================================
# 9. SHUTDOWN
# =============================================================================

class TestShutdown:
    def test_shutdown_after_run_does_not_raise(self):
        ki = _ki(SAAS_HTML_WITH_MAIN)
        execute_sync(ki, registry=registry, validator_check=validator.check)
        shutdown_sync()

    def test_double_shutdown_is_safe(self):
        shutdown_sync()
        shutdown_sync()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))