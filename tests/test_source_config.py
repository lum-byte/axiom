from __future__ import annotations

import json
from types import SimpleNamespace

from tag.crawler import source_config


def test_seed_domains_and_urls_come_from_dynamic_config(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "crawler_sources.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "custom",
                        "terms": ["neutrino"],
                        "domains": ["dynamic.example", "archive.example"],
                    }
                ],
                "search_templates": {
                    "dynamic.example": "https://dynamic.example/find/{query}",
                },
                "article_templates": {
                    "archive.example": "https://archive.example/article/{slug}",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(source_config.CONFIG_ENV, str(config_path))

    domains = source_config.seed_domains_for_query("latest neutrino result")
    assert domains == ["dynamic.example", "archive.example"]

    urls = source_config.source_urls_for_domains("latest neutrino result", domains, reason="test")
    assert urls[0]["url"] == "https://dynamic.example/find/latest%20neutrino%20result"
    assert source_config.domain_article_url("archive.example", "Latest Neutrino Result") == (
        "https://archive.example/article/Latest_Neutrino_Result"
    )


def test_source_domain_env_overrides_profiles(monkeypatch) -> None:
    monkeypatch.setenv(source_config.SOURCE_DOMAIN_ENV, "one.example, https://two.example/path ; one.example")
    assert source_config.configured_source_domains("ignored query") == ["one.example", "two.example"]


def test_clearance_policies_can_go_deep_without_dev_mode(monkeypatch) -> None:
    fetcher = SimpleNamespace(
        cl_state=SimpleNamespace(
            cl1_available=True,
            cl2_available=True,
            cl3_available=False,
            cl4_available=True,
        )
    )

    monkeypatch.delenv(source_config.CLEARANCE_LEVELS_ENV, raising=False)
    monkeypatch.setenv(source_config.CLEARANCE_POLICY_ENV, "standard")
    assert source_config.clearance_levels(fetcher, env_mode="") == [1]

    monkeypatch.setenv(source_config.CLEARANCE_POLICY_ENV, "deep")
    assert source_config.clearance_levels(fetcher, env_mode="") == [1, 2, 4]

    monkeypatch.setenv(source_config.CLEARANCE_POLICY_ENV, "max")
    assert source_config.clearance_levels(fetcher, env_mode="") == [1, 2, 3, 4]


def test_dev_clearance_policy_still_requires_dev_unless_overridden(monkeypatch) -> None:
    fetcher = SimpleNamespace(
        cl_state=SimpleNamespace(
            cl1_available=True,
            cl2_available=True,
            cl3_available=True,
            cl4_available=False,
        )
    )
    monkeypatch.setenv(source_config.CLEARANCE_POLICY_ENV, "dev")
    assert source_config.clearance_levels(fetcher, env_mode="") == [1]
    assert source_config.clearance_levels(fetcher, env_mode="dev") == [1, 2, 3]

    monkeypatch.setenv(source_config.CLEARANCE_REQUIRE_DEV_ENV, "false")
    assert source_config.clearance_levels(fetcher, env_mode="") == [1, 2, 3]
