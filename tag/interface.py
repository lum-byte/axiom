"""
tag/interface.py
================
The only public AXIOM runtime surface.

Commands:
    search | query
    fetch  | URL
    learn  | domain
    status |
    quit   |
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import html
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, List, Optional, Protocol
from urllib.parse import quote, urljoin, urlparse

from signal_kernel.contracts import (
    InterfaceRequest,
    InterfaceResponse,
    KernelInput,
    MAX_RAW_CONTENT_BYTES,
    RawFetchEvent,
    SignalExtractedEvent,
    StoreHealthEvent,
    SystemStatus,
    TopologyClassification,
    new_run_id,
)
from signal_kernel.pipeline import execute_sync
from signal_kernel.recipes import registry as recipe_registry
from signal_kernel.recipes import validator as recipe_validator
from tag.config import apply_environment_defaults, load_config
from tag.cold_start import ColdStart
from tag.runtime_paths import RuntimePathResolver
from tag.crawler.source_config import (
    clearance_levels as configured_clearance_levels,
    domain_article_url as configured_domain_article_url,
    configured_source_domains,
    domain_query_url as configured_domain_query_url,
    link_expansion_limit,
    max_search_sources,
)
from tag.crawler.swarm import AxiomCrawlSwarm, AxiomCrawlSwarmConfig
from tag.crawler.swarm_bridge import (
    crawl_config_from_plan,
    normalize_crawl_plan,
    parse_swarm_search_payload,
    plan_from_generic_talk,
)
from tag.dic import DirectContextInjector, DirectlyInjectContextAssembler, ExternalMCPAnchorClient, HybridFusionRanker, QueryExpansionEngine
from tag.index_daemon import IndexDaemon
from tag.integrity_sentinel import mark_block, start_once_per_login
from tag.search_cache import SearchResultCache
from tag.veritas import VeritasEngine


COMMAND_RE = re.compile(r"^\s*(search|fetch|learn|status|quit)\s*\|\s*(.*?)\s*$", re.IGNORECASE)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HEADER_TITLE_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#\s>]+)""", re.IGNORECASE)
SEARCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)
MAX_SEARCH_BLOCKS = 8
MAX_BLOCK_CHARS = 900
MIN_BLOCK_CHARS = 80


@dataclass(frozen=True)
class ParsedCommand:
    command: str
    payload: str


@dataclass(frozen=True)
class HistoryItem:
    command: str
    payload: str
    status: str
    run_id: str
    created_unix: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class InterfaceMetrics:
    handled: int = 0
    errors: int = 0
    accepted: int = 0
    empty: int = 0
    total_latency_ms: float = 0.0
    by_command: Dict[str, int] = field(default_factory=dict)

    def record(self, command: str, status: str, latency_ms: float) -> None:
        self.handled += 1
        self.total_latency_ms += latency_ms
        self.by_command[command] = self.by_command.get(command, 0) + 1
        if status == "error":
            self.errors += 1
        elif status == "accepted":
            self.accepted += 1
        elif status == "empty":
            self.empty += 1

    def to_dict(self) -> Dict[str, Any]:
        avg = self.total_latency_ms / self.handled if self.handled else 0.0
        return {
            "handled": self.handled,
            "errors": self.errors,
            "accepted": self.accepted,
            "empty": self.empty,
            "avg_latency_ms": avg,
            "by_command": dict(sorted(self.by_command.items())),
        }


@dataclass
class RuntimePathCheck:
    name: str
    path: str
    exists: bool
    is_dir: bool
    writable: bool
    env_var: str = ""
    env_matches: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RuntimeSnapshot:
    store_ready: bool
    cold_start_complete: bool
    index_daemon_ready: bool
    bus_started: bool
    bus_mode: str
    crawler_ready: bool
    learned_domains: int
    queued_work_items: int
    cached_documents: int
    daemon_status: Dict[str, Any]
    runtime_paths: Dict[str, str] = field(default_factory=dict)
    runtime_path_checks: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SearchDocument:
    url: str
    domain: str
    title: str
    topology_class: str
    classification_confidence: float
    fetch_mode: str
    status_code: int
    clean_text: str
    kernel_signal: str
    blocks: List[str]
    fetched_unix: int
    links: List[str] = field(default_factory=list)

    @property
    def searchable_text(self) -> str:
        return self.kernel_signal or self.clean_text


class FetcherBusBridge:
    """
    Compatibility bridge between fetcher.py's in-process event API and the
    canonical typed crawler bus.

    fetcher.py expects:
        subscribe(EventType, handler)
        await emit(event_instance)

    crawler_bus.py exposes:
        await emitter(topic, component, schema)
        await subscribe(topic, group, handler, schema)

    The bridge keeps fetcher's local synchronous expectations intact while
    forwarding emitted canonical events onto the real bus when possible.
    """

    def __init__(self, canonical_bus: Any, topic_registry: Dict[str, Any]) -> None:
        self._canonical_bus = canonical_bus
        self._subscriptions: Dict[type, List[Any]] = {}
        self._schema_to_topic = {schema: topic for topic, schema in topic_registry.items()}
        self._emitters: Dict[type, Any] = {}

    def subscribe(self, event_type: type, handler: Any) -> None:
        self._subscriptions.setdefault(event_type, []).append(handler)

    async def emit(self, event: Any) -> None:
        for handler in self._subscriptions.get(type(event), []):
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        topic = self._schema_to_topic.get(type(event))
        if topic is None:
            return
        emitter = self._emitters.get(type(event))
        if emitter is None:
            emitter = await self._canonical_bus.emitter(
                topic=topic,
                component="tag.interface.FetcherBusBridge",
                schema=type(event),
            )
            self._emitters[type(event)] = emitter
        await emitter.emit(event)


class AxiomRuntimeContext:
    """
    Internal lifetime manager for AXIOM runtime resources.

    It is intentionally not a public query engine.  The public boundary remains
    AxiomInterface/interface.py; this object owns process-local resources and
    shutdown order for store handles, index daemon state, and queued work.
    """

    def __init__(self, *, store_dir: Path = Path("store"), autostart: bool = False) -> None:
        mark_block("runtime.context.init")
        self.config = load_config()
        apply_environment_defaults(self.config)
        self.path_resolver = RuntimePathResolver(config=self.config)
        store_override = None if Path(store_dir) == Path("store") else Path(store_dir)
        self.paths = self.path_resolver.resolve(store_dir_override=store_override)
        self.paths.apply_environment(override=True)
        self.store_dir = self.paths.store_dir
        self.autostart = autostart
        self.cold_start = ColdStart(store_dir=self.store_dir, config=self.config)
        self.index_daemon: Optional[IndexDaemon] = None
        self.watchdog: Optional[Any] = None
        self.bus: Optional[Any] = None
        self.fetcher: Optional[Any] = None
        self.fetcher_bus_bridge: Optional[FetcherBusBridge] = None
        self.classifier: Optional[Any] = None
        self.sanitizer: Optional[Any] = None
        self.learned_domains: set[str] = set()
        self.queued_work: List[Dict[str, Any]] = []
        self.document_cache: Dict[str, SearchDocument] = {}
        self.search_cache = SearchResultCache(store_dir=self.store_dir, config=self.config)
        self.pending_crawl_plans: Dict[str, Dict[str, Any]] = {}
        self.runtime_dependency_errors: Dict[str, str] = {}
        self._dev_tor_process: Optional[subprocess.Popen[Any]] = None
        self.started = False
        self.start_result: Optional[Any] = None
        self._closing = False
        self._watchdog_registered = False

    async def __aenter__(self) -> "AxiomRuntimeContext":
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.close()

    async def start(self) -> None:
        mark_block("runtime.context.start")
        if self.started:
            return
        if self.autostart:
            self.start_result = await asyncio.to_thread(self.cold_start.run)
            if self.start_result.ok:
                self.index_daemon = IndexDaemon(store_dir=self.store_dir)
                await self.index_daemon.start_background_tasks()
                await self._ensure_store_watchdog_ready()
        self.started = True

    async def ensure_index_daemon(self) -> Optional[IndexDaemon]:
        mark_block("runtime.index_daemon.ensure")
        phase_store = self.store_dir / "phase_states.mmap"
        if self.index_daemon is None and phase_store.exists():
            self.index_daemon = IndexDaemon(store_dir=self.store_dir)
        if self.index_daemon is not None:
            await self.index_daemon.start_background_tasks()
        return self.index_daemon

    async def ensure_crawl_stack(self) -> None:
        mark_block("runtime.crawl_stack.ensure")
        self._ensure_runtime_paths_ready()
        self._ensure_bus_hmac_ready()
        if self.start_result is None:
            self.start_result = await asyncio.to_thread(self.cold_start.run)
            if self.start_result.ok and self.index_daemon is None:
                self.index_daemon = IndexDaemon(store_dir=self.store_dir)
                await self.index_daemon.start_background_tasks()

        await self._ensure_dev_tor_ready()

        from tag.crawler.fetcher import Fetcher
        from tag.crawler_bus import CrawlerBus, TOPIC_REGISTRY
        from tag.topology.sanitize import Sanitizer

        await self.start()
        await self._ensure_store_watchdog_ready()
        if self.bus is None:
            self.bus = CrawlerBus()
            await self.bus.start()
            mark_block("runtime.bus.ready")
        if self.fetcher is None:
            self.fetcher_bus_bridge = FetcherBusBridge(self.bus, TOPIC_REGISTRY)
            self.fetcher = Fetcher(bus=self.fetcher_bus_bridge, store_dir=self.store_dir)
        if hasattr(self.fetcher, "is_initialized") and not self.fetcher.is_initialized:
            await self.fetcher.initialize()
            mark_block("runtime.fetcher.ready")
        if self.classifier is None:
            try:
                from tag.topology.classifier import TopologyClassifier

                self.classifier = TopologyClassifier(
                    model_path=str(self.store_dir / "topology_router.pt"),
                    phase_states_path=str(self.store_dir / "phase_states.mmap"),
                )
                self.runtime_dependency_errors.pop("classifier", None)
            except Exception as exc:
                self.runtime_dependency_errors["classifier"] = f"{type(exc).__name__}: {exc}"
                self.classifier = False
        if self.sanitizer is None:
            self.sanitizer = Sanitizer()
            mark_block("runtime.sanitizer.ready")

    async def warm_status_stack(self) -> None:
        if not self.config.bool("runtime.status_warmup", True):
            return
        timeout_s = self.config.float("runtime.status_warmup_timeout_seconds", 5.0, low=0.1, high=60.0)
        try:
            await asyncio.wait_for(self._warm_status_stack_impl(), timeout=timeout_s)
            self.runtime_dependency_errors.pop("status_warmup", None)
        except asyncio.TimeoutError:
            self.runtime_dependency_errors["status_warmup"] = f"timed out after {timeout_s:.1f}s"
        except Exception as exc:
            self.runtime_dependency_errors["status_warmup"] = f"{type(exc).__name__}: {exc}"

    async def _warm_status_stack_impl(self) -> None:
        self._ensure_runtime_paths_ready()
        self._ensure_bus_hmac_ready()
        if self.start_result is None:
            self.start_result = await asyncio.to_thread(self.cold_start.run)
        if self.start_result is not None and self.start_result.ok:
            await self.ensure_index_daemon()
            if self.config.bool("runtime.status_start_watchdog", False):
                await self._ensure_store_watchdog_ready()
        if self.bus is None:
            from tag.crawler_bus import CrawlerBus

            self.bus = CrawlerBus()
            await self.bus.start()
        if self.fetcher is None:
            from tag.crawler.fetcher import Fetcher
            from tag.crawler_bus import TOPIC_REGISTRY

            self.fetcher_bus_bridge = FetcherBusBridge(self.bus, TOPIC_REGISTRY)
            self.fetcher = Fetcher(bus=self.fetcher_bus_bridge, store_dir=self.store_dir)
        if self.classifier is None:
            try:
                from tag.topology.classifier import TopologyClassifier

                self.classifier = TopologyClassifier(
                    model_path=str(self.store_dir / "topology_router.pt"),
                    phase_states_path=str(self.store_dir / "phase_states.mmap"),
                )
                self.runtime_dependency_errors.pop("classifier", None)
            except Exception as exc:
                self.runtime_dependency_errors["classifier"] = f"{type(exc).__name__}: {exc}"
                self.classifier = False
        if self.sanitizer is None:
            from tag.topology.sanitize import Sanitizer

            self.sanitizer = Sanitizer()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.fetcher is not None:
            await self.fetcher.shutdown()
            self.fetcher = None
            self.fetcher_bus_bridge = None
        self.search_cache.close()
        if self.bus is not None:
            await self.bus.stop()
            self.bus = None
        if self.watchdog is not None:
            await self.watchdog.stop()
            self.watchdog = None
            self._watchdog_registered = False
        if self._dev_tor_process is not None:
            self._dev_tor_process.terminate()
            try:
                self._dev_tor_process.wait(timeout=10.0)
            except Exception:
                self._dev_tor_process.kill()
            self._dev_tor_process = None
        if self.index_daemon is not None:
            if hasattr(self.index_daemon, "aclose"):
                await self.index_daemon.aclose()
            else:
                self.index_daemon.close()
            self.index_daemon = None
        self.classifier = None
        self.sanitizer = None
        self.started = False
        self._closing = False

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator["AxiomRuntimeContext"]:
        await self.start()
        try:
            yield self
        finally:
            await self.close()

    def enqueue(self, item: Dict[str, Any]) -> None:
        item.setdefault("created_unix", int(time.time()))
        self.queued_work.append(item)

    def _ensure_bus_hmac_ready(self) -> None:
        key = os.environ.get("AXIOM_BUS_HMAC_KEY", "")
        if key:
            return
        if self.env_mode in {"prod", "production", "release"}:
            return
        if not self.config.bool("bus.auto_dev_hmac", True):
            return
        os.environ["AXIOM_BUS_HMAC_KEY"] = hashlib.sha256(
            f"axiom-dev:{self.store_dir.resolve()}".encode("utf-8")
        ).hexdigest()

    def _ensure_runtime_paths_ready(self) -> None:
        self.paths = self.path_resolver.resolve(store_dir_override=self.store_dir)
        self.paths.ensure_base_dirs()
        self.paths.apply_environment(override=True)
        self.store_dir = self.paths.store_dir

    async def _ensure_store_watchdog_ready(self) -> None:
        if not self.config.bool("watchdog.enabled", True) or not self.config.bool("watchdog.start", True):
            return
        if self.watchdog is not None:
            return
        try:
            from tag.store_watchdog import StoreWatchdog
        except Exception as exc:
            self.runtime_dependency_errors["store_watchdog"] = f"{type(exc).__name__}: {exc}"
            return

        self.paths.ensure_base_dirs()
        for dirname in ("triggers", "staging", "dead_letters", "offline_queue"):
            (self.store_dir / dirname).mkdir(parents=True, exist_ok=True)
        watchdog = StoreWatchdog(store_root=self.store_dir)
        debounce = self.config.section("watchdog").get("debounce_ms", {})
        path_to_key = {
            "topology_router.pt": "topology_router_pt",
            "structural_layer.pt": "structural_layer_pt",
            "recipe_registry.mmap": "recipe_registry_mmap",
            "phase_states.mmap": "phase_states_mmap",
        }
        for path, key in path_to_key.items():
            async def handler(path: str = path) -> None:
                await self._handle_store_reload(path)

            configured_ms = debounce.get(key) if isinstance(debounce, dict) else None
            watchdog.register(path, handler, debounce_ms=int(configured_ms) if configured_ms is not None else None)
        try:
            await watchdog.start()
        except Exception as exc:
            self.runtime_dependency_errors["store_watchdog"] = f"{type(exc).__name__}: {exc}"
            return
        self.watchdog = watchdog
        self._watchdog_registered = True
        self.runtime_dependency_errors.pop("store_watchdog", None)

    async def _handle_store_reload(self, store_file: str) -> None:
        full_path = self.store_dir / store_file
        try:
            size = full_path.stat().st_size
            status = "changed"
            detail = "debounced store reload"
        except OSError as exc:
            size = 0
            status = "missing"
            detail = f"{type(exc).__name__}: {exc}"
        event = StoreHealthEvent(
            store_file=store_file,
            status=status,
            size_bytes=size,
            checksum_sha256=None,
            critical=status != "changed",
            detail=detail,
            run_id=str(new_run_id()),
        )
        daemon = await self.ensure_index_daemon()
        if daemon is not None:
            await daemon.dispatch(event)
        self.enqueue(
            {
                "type": "store_watchdog_reload",
                "store_file": store_file,
                "status": status,
                "size_bytes": size,
                "run_id": event.run_id,
            }
        )

    def remember_document(self, document: SearchDocument) -> None:
        self.document_cache[document.url] = document
        if len(self.document_cache) <= 128:
            return
        oldest_url = min(self.document_cache.items(), key=lambda item: item[1].fetched_unix)[0]
        self.document_cache.pop(oldest_url, None)

    def snapshot(self) -> RuntimeSnapshot:
        result = self.start_result
        daemon_status: Dict[str, Any] = {}
        if self.index_daemon is not None:
            daemon_status = self.index_daemon.status()
        bus_started, bus_mode = self._bus_status()
        path_checks = self._runtime_path_checks()
        errors = list(getattr(result, "errors", []) or [])
        errors.extend(
            f"{component}: {detail}"
            for component, detail in sorted(self.runtime_dependency_errors.items())
        )
        errors.extend(
            f"runtime path {check.name} is not writable: {check.path}"
            for check in path_checks
            if check.exists and check.is_dir and not check.writable
        )
        return RuntimeSnapshot(
            store_ready=self.store_dir.exists(),
            cold_start_complete=bool(result.ok) if result is not None else self.store_dir.exists(),
            index_daemon_ready=self.index_daemon is not None,
            bus_started=bus_started,
            bus_mode=bus_mode,
            crawler_ready=self.fetcher is not None and self.sanitizer is not None,
            learned_domains=len(self.learned_domains),
            queued_work_items=len(self.queued_work),
            cached_documents=len(self.document_cache),
            daemon_status=daemon_status,
            runtime_paths=self.paths.to_dict(),
            runtime_path_checks=[check.to_dict() for check in path_checks],
            warnings=list(getattr(result, "warnings", []) or []),
            errors=errors,
        )

    def _bus_status(self) -> tuple[bool, str]:
        if self.bus is None:
            return False, "unstarted"
        try:
            health = self.bus.health()
            return bool(getattr(health, "started", True)), str(getattr(health, "mode", "started"))
        except Exception:
            return True, "started"

    def _runtime_path_checks(self) -> List[RuntimePathCheck]:
        candidates: List[tuple[str, Path, str]] = [
            ("root", self.paths.root, "AXIOM_ROOT"),
            ("store_dir", self.paths.store_dir, "AXIOM_STORE_DIR"),
            ("runtime_root", self.paths.runtime_root, "AXIOM_RUNTIME_ROOT"),
            ("tmp_dir", self.paths.tmp_dir, "AXIOM_TMP_DIR"),
            ("release_root", self.paths.release_root, "AXIOM_RELEASE_ROOT"),
            ("search_cache_path", self.paths.search_cache_path, "AXIOM_SEARCH_CACHE_PATH"),
            ("search_cache_index_path", self.paths.search_cache_index_path, "AXIOM_SEARCH_CACHE_INDEX_PATH"),
            ("html_snapshot_dir", self.paths.html_snapshot_dir, "AXIOM_HTML_SNAPSHOT_DIR"),
            ("fetch_staging_path", self.paths.fetch_staging_path, "AXIOM_FETCH_STAGING_PATH"),
            ("tor_work_dir", self.paths.tor_work_dir, "AXIOM_TOR_WORK_DIR"),
        ]
        checks = [self._path_check(name, path, env_var) for name, path, env_var in candidates]
        for index, candidate in enumerate(self.paths.native_library_candidates(), start=1):
            checks.append(self._path_check(f"native_library_candidate_{index}", candidate, ""))
        return checks

    @staticmethod
    def _path_check(name: str, path: Path, env_var: str) -> RuntimePathCheck:
        env_value = os.environ.get(env_var, "") if env_var else ""
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        check_target = path if is_dir else path.parent
        writable = check_target.exists() and os.access(check_target, os.W_OK)
        env_matches = True
        if env_var and env_value:
            env_matches = Path(env_value).expanduser().resolve(strict=False) == path.resolve(strict=False)
        return RuntimePathCheck(
            name=name,
            path=str(path),
            exists=exists,
            is_dir=is_dir,
            writable=writable,
            env_var=env_var,
            env_matches=env_matches,
        )

    @property
    def env_mode(self) -> str:
        return os.environ.get("AXIOM_ENV", "").strip().lower()

    @property
    def is_dev_mode(self) -> bool:
        return self.env_mode == "dev"

    async def _ensure_dev_tor_ready(self) -> None:
        if not self.is_dev_mode:
            return
        if await self._probe_local_port(9050):
            self.runtime_dependency_errors.pop("tor", None)
            return
        tor_exe = self._resolve_dev_tor_exe()
        if tor_exe is None:
            self.runtime_dependency_errors["tor"] = (
                "Tor runtime missing: set AXIOM_TOR_EXE or configure paths.tor_bundle_root "
                "for the expert bundle"
            )
            return
        runtime_root = self.paths.tor_work_dir
        data_dir = self.paths.tor_data_dir
        tor_data_root = self._resolve_dev_tor_data_root(tor_exe)
        runtime_root.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        torrc = self.paths.torrc_path
        torrc_lines = [
            f'DataDirectory "{data_dir.as_posix()}"',
            "SocksPort 127.0.0.1:9050",
            "ControlPort 127.0.0.1:9051",
            "CookieAuthentication 0",
            "AvoidDiskWrites 1",
            "Log notice stdout",
        ]
        geoip = tor_data_root / "geoip"
        geoip6 = tor_data_root / "geoip6"
        if geoip.exists():
            torrc_lines.insert(4, f'GeoIPFile "{geoip.as_posix()}"')
        if geoip6.exists():
            torrc_lines.insert(5, f'GeoIPv6File "{geoip6.as_posix()}"')
        torrc.write_text("\n".join(torrc_lines) + "\n", encoding="utf-8")
        if self._dev_tor_process is None or self._dev_tor_process.poll() is not None:
            startupinfo: Optional[subprocess.STARTUPINFO] = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            env = os.environ.copy()
            if os.name != "nt":
                existing_ld = env.get("LD_LIBRARY_PATH", "")
                bundle_lib = str(tor_exe.parent)
                env["LD_LIBRARY_PATH"] = f"{bundle_lib}:{existing_ld}" if existing_ld else bundle_lib
            try:
                self._dev_tor_process = subprocess.Popen(
                    [str(tor_exe), "-f", str(torrc)],
                    cwd=str(tor_exe.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    startupinfo=startupinfo,
                    env=env,
                )
            except OSError as exc:
                self.runtime_dependency_errors["tor"] = f"Tor launch failed: {type(exc).__name__}: {exc}"
                self._dev_tor_process = None
                return
        for _ in range(40):
            if self._dev_tor_process is not None and self._dev_tor_process.poll() is not None:
                self.runtime_dependency_errors["tor"] = (
                    f"Tor exited during startup with code {self._dev_tor_process.returncode}"
                )
                self._dev_tor_process = None
                return
            socks_ok = await self._probe_local_port(9050)
            control_ok = await self._probe_local_port(9051)
            if socks_ok and control_ok:
                self.runtime_dependency_errors.pop("tor", None)
                return
            await asyncio.sleep(0.5)
        self.runtime_dependency_errors["tor"] = "Tor startup timed out waiting for ports 9050/9051"

    async def _probe_local_port(self, port: int) -> bool:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except Exception:
            return False
        writer.close()
        await writer.wait_closed()
        return True

    def _resolve_dev_tor_exe(self) -> Optional[Path]:
        for candidate in self.paths.tor_executable_candidates(os_name=os.name):
            if candidate.exists() and self._is_usable_tor_executable(candidate):
                return candidate
        return None

    def _resolve_dev_tor_data_root(self, tor_exe: Path) -> Path:
        for candidate in self.paths.tor_data_candidates(tor_exe):
            if candidate.exists():
                return candidate
        return self.paths.tor_data_dir

    def _is_usable_tor_executable(self, candidate: Path) -> bool:
        if os.name == "nt":
            return candidate.is_file()
        return candidate.is_file() and candidate.suffix.lower() != ".exe" and os.access(candidate, os.X_OK)


class QueryOrchestrator:
    """
    Internal command router under the single public interface.

    `search |` remains AXIOM TAG routing over learned source priority and queued
    frontier expansion.  The final `_synthesize()` method is the single
    inference point; the current implementation is deterministic until an LLM
    provider is configured.
    """

    def __init__(self, runtime: AxiomRuntimeContext) -> None:
        self.runtime = runtime

    async def handle(self, req: InterfaceRequest) -> InterfaceResponse:
        if req.query_type == "STATUS":
            return await self._status(req)
        if req.query_type == "QUIT":
            await self.runtime.close()
            return InterfaceResponse(run_id=req.run_id, status="ok", message="quit accepted", data={"quit": True})
        if req.query_type == "LEARN":
            return self._learn(req)
        if req.query_type == "FETCH":
            return await self._fetch(req)
        if req.query_type == "SEARCH":
            return await self._search(req)
        return InterfaceResponse(run_id=req.run_id, status="error", message="unknown command", data={})

    async def _status(self, req: InterfaceRequest) -> InterfaceResponse:
        await self.runtime.warm_status_stack()
        snapshot = self.runtime.snapshot()
        status = SystemStatus(
            run_id=req.run_id,
            bus_started=snapshot.bus_started,
            bus_mode=snapshot.bus_mode,
            store_ready=snapshot.store_ready,
            index_daemon_ready=snapshot.index_daemon_ready,
            cold_start_complete=snapshot.cold_start_complete,
            learned_domains=snapshot.learned_domains,
            queued_work_items=snapshot.queued_work_items,
        )
        return InterfaceResponse(
            run_id=req.run_id,
            status="ok",
            message="status",
            data={
                **status.__dict__,
                "daemon_status": snapshot.daemon_status,
                "runtime_paths": snapshot.runtime_paths,
                "runtime_path_checks": snapshot.runtime_path_checks,
                "crawler_ready": snapshot.crawler_ready,
                "cached_documents": snapshot.cached_documents,
                "warnings": snapshot.warnings,
                "errors": snapshot.errors,
            },
        )

    def _learn(self, req: InterfaceRequest) -> InterfaceResponse:
        domain = AxiomInterface.normalize_domain(req.payload)
        if not domain:
            return InterfaceResponse(run_id=req.run_id, status="error", message="learn requires a domain", data={"payload": req.payload})
        self.runtime.learned_domains.add(domain)
        self.runtime.enqueue({"type": "learn", "domain": domain, "run_id": req.run_id})
        return InterfaceResponse(run_id=req.run_id, status="accepted", message="learning queued", data={"domain": domain})

    async def _fetch(self, req: InterfaceRequest) -> InterfaceResponse:
        url = req.payload.strip()
        if not AxiomInterface.valid_http_url(url):
            return InterfaceResponse(run_id=req.run_id, status="error", message="fetch requires http(s) URL", data={"url": url})
        document = await self._fetch_document(url, req.run_id, reason="explicit_fetch")
        if document is None:
            return InterfaceResponse(
                run_id=req.run_id,
                status="error",
                message="fetch failed",
                data={"url": url, "fetch_mode": "static"},
            )
        self.runtime.enqueue({"type": "fetch", "url": url, "run_id": req.run_id})
        return InterfaceResponse(
            run_id=req.run_id,
            status="ok",
            message=f"fetched {document.url}",
            data={
                "url": document.url,
                "title": document.title,
                "fetch_mode": document.fetch_mode,
                "status_code": document.status_code,
                "topology_class": document.topology_class,
                "blocks": document.blocks[:3],
            },
        )

    async def _search(self, req: InterfaceRequest) -> InterfaceResponse:
        query, inline_crawl_plan = parse_swarm_search_payload(req.payload.strip())
        crawl_plan = inline_crawl_plan or self.runtime.pending_crawl_plans.pop(req.run_id, None)
        if crawl_plan is not None:
            crawl_plan = normalize_crawl_plan(crawl_plan, default_query=query)
            query = query or str(crawl_plan.get("query", "")).strip()
        if not query:
            return InterfaceResponse(run_id=req.run_id, status="error", message="query is empty", data={})
        force_recheck = bool(crawl_plan and crawl_plan.get("recheck"))
        cache_key = self.runtime.search_cache.build_key(query=query, crawl_plan=crawl_plan)
        if not force_recheck:
            cached = self.runtime.search_cache.get(cache_key)
            if cached is not None:
                return self.runtime.search_cache.response_from_entry(cached, run_id=req.run_id)
        swarm_config = crawl_config_from_plan(crawl_plan)
        expansion = QueryExpansionEngine(config=self.runtime.config).expand(
            query,
            requested_limit=int(crawl_plan.get("expansion_count", 0) if crawl_plan else 0),
        )
        swarm_config = self._scale_swarm_for_expansion(swarm_config, expansion.effective_limit)
        candidates = self._candidate_sources(query, crawl_plan=crawl_plan)
        candidates = self._merge_expanded_candidates(query, candidates, expansion)
        self._enqueue_tool_assist_plan(query, candidates, req.run_id)
        use_mcp_anchors = bool(crawl_plan) or getattr(expansion, "effective_limit", 0) > 0 or self.runtime.config.bool("mcp.anchor_always", False)
        mcp_anchor_task = (
            asyncio.create_task(self._collect_mcp_anchor_documents(query, req.run_id, expansion))
            if use_mcp_anchors
            else None
        )
        if not candidates:
            mcp_anchor_documents = await mcp_anchor_task if mcp_anchor_task is not None else []
            if mcp_anchor_documents:
                ranked_blocks = self._rank_documents(query, mcp_anchor_documents)
                ranked_blocks = HybridFusionRanker(config=self.runtime.config).rank(query, ranked_blocks)
                veritas = await VeritasEngine(config=self.runtime.config).classify(query, ranked_blocks)
                dic_context = DirectlyInjectContextAssembler(config=self.runtime.config).assemble(
                    query=query,
                    ranked_blocks=ranked_blocks,
                    expansion=expansion,
                    veritas=veritas,
                )
                injected = DirectContextInjector().format(dic_context)
                answer = {
                    "text": dic_context.answer,
                    "structured": dic_context.structured_answer,
                    "source": dic_context.citations[0] if dic_context.citations else None,
                    "sources": list(dic_context.structured_answer.get("citation_spine") or dic_context.citations[:8]),
                }
                response = InterfaceResponse(
                    run_id=req.run_id,
                    status="ok",
                    message=dic_context.answer,
                    data={
                        "query": query,
                        "sources": [],
                        "blocks": ranked_blocks,
                        "answer": answer,
                        "direct_inject_context": injected.json_payload,
                        "injection_text": injected.text,
                        "veritas": veritas,
                        "query_expansion": expansion.to_dict(),
                        "mcp_anchor_documents": len(mcp_anchor_documents),
                        "single_inference_point": "tag.interface.QueryOrchestrator._synthesize",
                        "search_engine": False,
                        "crawl_swarm": self._crawl_swarm_summary(crawl_plan, swarm_config),
                        "cache": {"hit": False, "key": cache_key},
                    },
                )
                self.runtime.search_cache.put_response(cache_key, response)
                return response
            self.runtime.enqueue({"type": "learn_from_query", "query": query, "run_id": req.run_id})
            return InterfaceResponse(
                run_id=req.run_id,
                status="empty",
                message="no learned topology candidates; learning queued",
                data={"query": query, "sources": []},
            )
        documents = await self._collect_documents(query, req.run_id, candidates, swarm_config=swarm_config)
        mcp_anchor_documents = await mcp_anchor_task if mcp_anchor_task is not None else []
        if mcp_anchor_documents:
            documents = [*mcp_anchor_documents, *documents]
        ranked_blocks = self._rank_documents(query, documents)
        ranked_blocks = HybridFusionRanker(config=self.runtime.config).rank(query, ranked_blocks)
        veritas = await VeritasEngine(config=self.runtime.config).classify(query, ranked_blocks)
        dic_context = DirectlyInjectContextAssembler(config=self.runtime.config).assemble(
            query=query,
            ranked_blocks=ranked_blocks,
            expansion=expansion,
            veritas=veritas,
        )
        injected = DirectContextInjector().format(dic_context)
        answer = {
            "text": dic_context.answer,
            "structured": dic_context.structured_answer,
            "source": dic_context.citations[0] if dic_context.citations else None,
            "sources": list(dic_context.structured_answer.get("citation_spine") or dic_context.citations[:8]),
        }
        signal = dic_context.answer
        response = InterfaceResponse(
            run_id=req.run_id,
            status="ok",
            message=signal,
            data={
                "query": query,
                "sources": candidates,
                "blocks": ranked_blocks,
                "answer": answer,
                "direct_inject_context": injected.json_payload,
                "injection_text": injected.text,
                "veritas": veritas,
                "query_expansion": expansion.to_dict(),
                "single_inference_point": "tag.interface.QueryOrchestrator._synthesize",
                "search_engine": False,
                "crawl_swarm": self._crawl_swarm_summary(crawl_plan, swarm_config),
                "cache": {"hit": False, "key": cache_key},
            },
        )
        self.runtime.search_cache.put_response(cache_key, response)
        return response

    def _scale_swarm_for_expansion(self, config: AxiomCrawlSwarmConfig, expansion_count: int) -> AxiomCrawlSwarmConfig:
        if expansion_count <= 0:
            return config
        total_cap = self.runtime.config.int("dic.max_total_crawlers", 100, low=1, high=5000)
        hard_cap = self.runtime.config.int("dic.hard_total_crawlers", 500, low=1, high=10000)
        requested = max(1, config.requested_worker_count or config.worker_count)
        desired = requested * max(1, expansion_count + 1)
        worker_count = max(1, min(hard_cap, max(config.worker_count, min(total_cap, desired))))
        return replace(config, worker_count=worker_count, max_worker_count=max(config.max_worker_count, worker_count))

    def _merge_expanded_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        expansion: Any,
    ) -> List[Dict[str, Any]]:
        if getattr(expansion, "effective_limit", 0) <= 0:
            return candidates
        merged = list(candidates)
        seen = {str(candidate.get("url") or "") for candidate in merged}
        max_sources = self._max_search_sources()
        for directive in expansion.directives[1:]:
            for candidate in self._candidate_sources(directive.query, crawl_plan=None):
                url = str(candidate.get("url") or "")
                if not url or url in seen:
                    continue
                seen.add(url)
                enriched = dict(candidate)
                enriched["reason"] = f"dic_expansion_{directive.query_type.lower()}_{candidate.get('reason', 'source')}"
                enriched["expansion_query"] = directive.query
                enriched["query_type"] = directive.query_type
                merged.append(enriched)
                if len(merged) >= max_sources:
                    return merged
        return merged

    def _candidate_sources(self, query: str, *, crawl_plan: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        terms = {term for term in re.split(r"\W+", query.lower()) if term}
        domains: Dict[str, str] = {}
        if crawl_plan is not None:
            for domain in crawl_plan.get("seed_domains", []):
                normalized = AxiomInterface.normalize_domain(str(domain))
                if normalized:
                    domains[normalized] = "swarm_plan"
        for domain in self.runtime.learned_domains:
            domains.setdefault(domain, "learned")
        for domain in self._source_seed_domains(query):
            domains.setdefault(domain, "source_seed")

        ranked: List[tuple[int, int, str, str]] = []
        source_priority = {"swarm_plan": 0, "learned": 1, "source_seed": 2}
        for domain, source_kind in domains.items():
            score = sum(1 for term in terms if term in domain)
            source_penalty = source_priority.get(source_kind, 3)
            ranked.append((source_penalty, -score, domain, source_kind))
        ranked.sort()
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        slug = self._query_slug(query)
        quoted_query = quote(query)
        max_sources = self._max_search_sources()

        def add_candidate(url: str, domain: str, reason: str, *, cached: bool = False, seeded: bool = False) -> None:
            if url in seen:
                return
            seen.add(url)
            candidates.append({"url": url, "domain": domain, "reason": reason, "cached": cached, "seeded": seeded})

        if crawl_plan is not None:
            if self._prefer_article_guess(query):
                for domain in crawl_plan.get("seed_domains", []):
                    normalized = AxiomInterface.normalize_domain(str(domain))
                    if not normalized:
                        continue
                    article_url = configured_domain_article_url(normalized, slug)
                    if article_url:
                        add_candidate(article_url, normalized, "swarm_plan_article_guess", seeded=True)
            for source in crawl_plan.get("source_urls", []):
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url", "")).strip()
                domain = AxiomInterface.normalize_domain(str(source.get("domain") or url))
                if not url or not domain:
                    continue
                add_candidate(
                    url,
                    domain,
                    str(source.get("reason") or "swarm_bridge_source"),
                    cached=bool(source.get("cached", False)),
                    seeded=bool(source.get("seeded", True)),
                )

        for _, _, domain, source_kind in ranked:
            seeded = source_kind in {"source_seed", "swarm_plan"}
            for document in self.runtime.document_cache.values():
                if document.domain == domain:
                    add_candidate(document.url, domain, "cache", cached=True, seeded=seeded)
            article_url = configured_domain_article_url(domain, slug)
            if article_url:
                add_candidate(
                    article_url,
                    domain,
                    self._source_reason(source_kind, "article_guess"),
                    seeded=seeded,
                )
            site_search = self._domain_query_url(domain)
            if site_search:
                add_candidate(
                    site_search.format(query=quoted_query),
                    domain,
                    self._source_reason(source_kind, "site_search"),
                    seeded=seeded,
                )
            add_candidate(
                f"https://{domain}/",
                domain,
                self._source_reason(source_kind, "root"),
                seeded=seeded,
            )
            if len(candidates) >= max_sources:
                break
        return candidates[:max_sources]

    def _source_reason(self, source_kind: str, suffix: str) -> str:
        if source_kind == "swarm_plan":
            return f"swarm_plan_{suffix}"
        if source_kind == "source_seed":
            return f"source_seed_{suffix}"
        return f"learned_domain_{suffix}"

    def _source_seed_domains(self, query: str = "") -> List[str]:
        return configured_source_domains(query)

    def _domain_query_url(self, domain: str) -> str:
        return configured_domain_query_url(domain)

    def _max_search_sources(self) -> int:
        return max_search_sources()

    def _query_terms(self, query: str) -> List[str]:
        raw_terms = [term for term in re.split(r"\W+", query.lower()) if term]
        strong_terms = [term for term in raw_terms if len(term) > 1 and term not in SEARCH_STOPWORDS]
        return strong_terms or raw_terms

    def _query_slug(self, query: str) -> str:
        terms = self._query_terms(query)
        if not terms:
            return ""
        if terms[0] in {"what", "who", "where", "when", "why", "how"} and len(terms) > 1:
            terms = terms[1:]
        return " ".join(term.capitalize() for term in terms[:8])

    def _prefer_article_guess(self, query: str) -> bool:
        lowered = re.sub(r"\s+", " ", query.lower()).strip()
        return lowered.startswith(
            (
                "what is ",
                "what are ",
                "who is ",
                "who was ",
                "where is ",
                "when was ",
            )
        )

    async def _collect_documents(
        self,
        query: str,
        run_id: str,
        candidates: List[Dict[str, Any]],
        *,
        swarm_config: Optional[AxiomCrawlSwarmConfig] = None,
    ) -> List[SearchDocument]:
        mark_block("crawler.swarm.collect")
        async def fetch_candidate(candidate: Dict[str, Any]) -> Optional[SearchDocument]:
            cached = self.runtime.document_cache.get(candidate["url"])
            if cached is not None:
                return cached
            fetch_kwargs: Dict[str, Any] = {"reason": candidate["reason"]}
            try:
                if "query" in inspect.signature(self._fetch_document).parameters:
                    fetch_kwargs["query"] = query
            except (TypeError, ValueError):
                pass
            return await self._fetch_document(candidate["url"], run_id, **fetch_kwargs)

        active_config = swarm_config or AxiomCrawlSwarmConfig.from_env()

        async def fetch_batch(batch: List[Dict[str, Any]]) -> List[Optional[SearchDocument]]:
            raw_results = await asyncio.gather(
                *(fetch_candidate(candidate) for candidate in batch[: active_config.worker_count]),
                return_exceptions=True,
            )
            documents: List[Optional[SearchDocument]] = []
            for candidate, result in zip(batch, raw_results):
                if isinstance(result, BaseException):
                    self.runtime.enqueue(
                        {
                            "type": "fetch_exception",
                            "url": candidate.get("url"),
                            "reason": candidate.get("reason"),
                            "error": f"{type(result).__name__}: {result}",
                            "run_id": run_id,
                        }
                    )
                    documents.append(None)
                else:
                    documents.append(result)
            return documents

        swarm = AxiomCrawlSwarm(
            fetch_document=fetch_candidate,
            fetch_batch=fetch_batch,
            rank_documents=lambda documents: self._rank_documents(query, documents),
            expand_document=lambda document: self._expand_document_candidates(document, query),
            config=active_config,
        )
        swarm_result = await swarm.collect(candidates)
        documents = list(swarm_result.documents)
        self.runtime.enqueue(
            {
                "type": "crawl_swarm_complete",
                "run_id": run_id,
                **swarm_result.telemetry,
            }
        )
        documents.sort(
            key=lambda item: (
                item.status_code >= 400,
                -item.classification_confidence,
                -item.fetched_unix,
            )
        )
        return documents

    def _crawl_swarm_summary(
        self,
        crawl_plan: Optional[Dict[str, Any]],
        swarm_config: AxiomCrawlSwarmConfig,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "worker_count": swarm_config.worker_count,
            "requested_worker_count": swarm_config.requested_worker_count,
            "max_worker_count": swarm_config.max_worker_count,
            "target_documents": swarm_config.target_documents,
            "max_waves": swarm_config.max_waves,
            "early_stop_score": swarm_config.early_stop_score,
            "one_worker_per_site": True,
        }
        if crawl_plan is None:
            summary["watermark"] = "axiom.default.open_web"
            return summary
        summary.update(
            {
                "watermark": crawl_plan.get("watermark"),
                "intent": crawl_plan.get("intent"),
                "seed_domains": list(crawl_plan.get("seed_domains", []))[:24],
                "source_urls": len(crawl_plan.get("source_urls", [])),
                "constraints": crawl_plan.get("constraints", {}),
            }
        )
        return summary

    async def _collect_mcp_anchor_documents(self, query: str, run_id: str, expansion: Any) -> List[SearchDocument]:
        if not self.runtime.config.bool("mcp.enabled", True) or not self.runtime.config.bool("mcp.anchor_process_enabled", True):
            return []
        max_anchor_results = self.runtime.config.int("mcp.max_anchor_results", 8, low=1, high=50)
        anchor_query_count = 1
        if getattr(expansion, "effective_limit", 0) > 0:
            anchor_query_count = min(3, max(1, expansion.effective_limit + 1))
        queries = list(getattr(expansion, "queries", []) or [query])[:anchor_query_count]
        client = ExternalMCPAnchorClient(config=self.runtime.config)

        async def fetch_anchor_query(anchor_query: str) -> List[Dict[str, Any]]:
            try:
                return await client.fetch_anchor_blocks(anchor_query, limit=max_anchor_results)
            except Exception as exc:  # noqa: BLE001 - MCP anchors are additive, not fatal
                self.runtime.enqueue(
                    {
                        "type": "mcp_anchor_error",
                        "query": anchor_query,
                        "error": f"{type(exc).__name__}: {exc}",
                        "run_id": run_id,
                    }
                )
                return []

        batches = await asyncio.gather(*(fetch_anchor_query(anchor_query) for anchor_query in queries), return_exceptions=False)
        blocks = []
        seen_urls: set[str] = set()
        for batch in batches:
            for block in batch:
                url = str(block.get("url") or "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                blocks.append(block)
        documents: List[SearchDocument] = []
        for block in blocks[: max_anchor_results * len(queries)]:
            text = str(block.get("text") or "").strip()
            url = str(block.get("url") or "")
            if not text or not url:
                continue
            document = SearchDocument(
                url=url,
                domain=AxiomInterface.normalize_domain(str(block.get("domain") or url)),
                title=str(block.get("title") or block.get("domain") or url),
                topology_class=str(block.get("topology_class") or "MCP_ANCHOR"),
                classification_confidence=float(block.get("classification_confidence") or 1.0),
                fetch_mode=str(block.get("fetch_mode") or "mcp"),
                status_code=200,
                clean_text=text,
                kernel_signal=text,
                blocks=self._split_text_blocks(text),
                fetched_unix=int(time.time()),
                links=[],
            )
            self.runtime.remember_document(document)
            documents.append(document)
        if documents:
            self.runtime.enqueue(
                {
                    "type": "mcp_anchor_documents",
                    "query": query,
                    "document_count": len(documents),
                    "queries": queries,
                    "run_id": run_id,
                }
            )
        return documents

    def _enqueue_tool_assist_plan(self, query: str, candidates: List[Dict[str, Any]], run_id: str) -> None:
        if os.environ.get("AXIOM_TOOL_ASSIST", "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        try:
            from tag.tools_bridge import ToolsBridge

            bridge = ToolsBridge(emit_bus=False)
            plan = bridge.assist_plan_for_query(
                query,
                candidate_urls=[str(candidate.get("url", "")) for candidate in candidates[:10]],
            )
        except Exception as exc:
            self.runtime.enqueue(
                {
                    "type": "tool_assist_unavailable",
                    "query": query,
                    "error": f"{type(exc).__name__}: {exc}",
                    "run_id": run_id,
                }
            )
            return
        self.runtime.enqueue(
            {
                "type": "tool_assist_plan",
                "query": query,
                "run_id": run_id,
                "plan": plan,
            }
        )

    async def _fetch_document(self, url: str, run_id: str, *, reason: str, query: str = "") -> Optional[SearchDocument]:
        mark_block("crawler.fetch_document")
        cached = self.runtime.document_cache.get(url)
        if cached is not None:
            return cached
        await self.runtime.ensure_crawl_stack()
        assert self.runtime.fetcher is not None
        assert self.runtime.classifier is not None
        assert self.runtime.sanitizer is not None
        attempted_levels: List[int] = []
        raw_event: Optional[RawFetchEvent] = None
        for cl_level in self._clearance_levels():
            attempted_levels.append(cl_level)
            fetch_single = self.runtime.fetcher.fetch_single
            kwargs: Dict[str, Any] = {
                "url": url,
                "cl_level": cl_level,
                "topology_hint": "GENERIC_HTML",
            }
            try:
                if "dedupe" in inspect.signature(fetch_single).parameters:
                    kwargs["dedupe"] = False
            except (TypeError, ValueError):
                pass
            raw_event = await fetch_single(**kwargs)
            if raw_event is not None:
                break
        if raw_event is None:
            self.runtime.enqueue(
                {
                    "type": "fetch_failed",
                    "url": url,
                    "reason": reason,
                    "clearance_attempts": attempted_levels,
                    "run_id": run_id,
                }
            )
            return None
        classification = await self._classify_fetch(raw_event, run_id)
        readable_text = self._extract_readable_text(raw_event.raw_bytes, raw_event.headers)
        clean_result = self.runtime.sanitizer.process(raw_event.raw_bytes)
        clean_text = clean_result.text.strip() if clean_result.ok else ""
        kernel_signal = await self._run_kernel(raw_event, classification, run_id)
        signal_text = self._select_signal_text(query=query, candidates=[kernel_signal, clean_text, readable_text])
        if not clean_text and readable_text:
            clean_text = readable_text
        blocks = self._split_text_blocks(signal_text)
        document = SearchDocument(
            url=raw_event.url,
            domain=AxiomInterface.normalize_domain(raw_event.url),
            title=self._extract_title(raw_event),
            topology_class=classification.topology_class,
            classification_confidence=classification.confidence,
            fetch_mode=str(getattr(raw_event.fetch_mode, "value", raw_event.fetch_mode)),
            status_code=raw_event.status_code,
            clean_text=clean_text,
            kernel_signal=kernel_signal,
            blocks=blocks,
            fetched_unix=int(time.time()),
            links=self._extract_links(raw_event.url, raw_event.raw_bytes),
        )
        self.runtime.remember_document(document)
        self.runtime.enqueue(
            {
                "type": "fetched_document",
                "url": raw_event.url,
                "reason": reason,
                "topology_class": document.topology_class,
                "clearance_attempts": attempted_levels,
                "effective_fetch_mode": document.fetch_mode,
                "run_id": run_id,
            }
        )
        await self._dispatch_signal_event(document, run_id)
        return document

    def _clearance_levels(self) -> List[int]:
        return configured_clearance_levels(self.runtime.fetcher, env_mode=self.runtime.env_mode)

    async def _classify_fetch(self, raw_event: RawFetchEvent, run_id: str) -> TopologyClassification:
        if self.runtime.classifier in (None, False):
            return TopologyClassification(
                topology_class=raw_event.topology_hint or "GENERIC_HTML",
                confidence=0.0,
                classification_path="fallback",
                signals_used={"fallback": "tag.interface"},
                latency_ms=0.0,
                run_id=run_id,
            )
        try:
            from tag.topology.classifier import ClassifierInput

            return await self.runtime.classifier.classify(
                ClassifierInput(
                    url=raw_event.url,
                    headers={str(key).lower(): str(value) for key, value in raw_event.headers.items()},
                    content_prefix=raw_event.raw_bytes[:65536],
                    response_code=raw_event.status_code,
                    run_id=run_id,
                )
            )
        except Exception:
            return TopologyClassification(
                topology_class=raw_event.topology_hint or "GENERIC_HTML",
                confidence=0.0,
                classification_path="fallback",
                signals_used={"fallback": "tag.interface"},
                latency_ms=0.0,
                run_id=run_id,
            )

    async def _run_kernel(
        self,
        raw_event: RawFetchEvent,
        classification: TopologyClassification,
        run_id: str,
    ) -> str:
        content_bytes = raw_event.raw_bytes[:MAX_RAW_CONTENT_BYTES]
        raw_content = content_bytes.decode("utf-8", errors="replace").strip()
        if not raw_content:
            return ""
        content_type_header = str(raw_event.headers.get("content-type", "")).lower()
        content_type = "json" if "json" in content_type_header else "html"
        kernel_input = KernelInput(
            raw_content=raw_content,
            topology_class=classification.topology_class or "GENERIC_HTML",
            intent_vector_hash=hashlib.sha256(f"{raw_event.url}|{classification.topology_class}".encode("utf-8")).hexdigest(),
            content_type=content_type,
            source_url=raw_event.url,
            run_id=run_id,
        )
        try:
            result = await asyncio.to_thread(
                execute_sync,
                kernel_input,
                registry=recipe_registry,
                validator_check=recipe_validator.check,
            )
        except Exception:
            return ""
        return result.clean_signal.strip() if not result.extraction_empty else ""

    async def _dispatch_signal_event(self, document: SearchDocument, run_id: str) -> None:
        daemon = await self.runtime.ensure_index_daemon()
        if daemon is None:
            return
        text = document.searchable_text.strip()
        if not text:
            return
        token_count = max(1, len(text.split()))
        zone_count = max(1, len(document.blocks))
        byte_count = max(1, len(text.encode("utf-8")))
        density = min(1.0, byte_count / max(1.0, float(document.status_code + byte_count)))
        await daemon.dispatch(
            SignalExtractedEvent(
                url=document.url,
                topology_class=document.topology_class,
                signal_type="prose",
                byte_count=byte_count,
                token_count=token_count,
                signal_density=max(0.05, density),
                zone_count=zone_count,
                source_component="tag.interface",
                run_id=run_id,
            )
        )

    def _expand_document_candidates(self, document: SearchDocument, query: str) -> List[Dict[str, Any]]:
        terms = set(self._query_terms(query))
        candidates: List[Dict[str, Any]] = []
        limit = self._link_expansion_limit()
        for link in document.links:
            domain = AxiomInterface.normalize_domain(link)
            if not domain or domain == document.domain:
                continue
            url_text = link.lower()
            score = sum(1 for term in terms if term in url_text)
            if score == 0 and len(candidates) >= max(2, limit // 3):
                continue
            candidates.append(
                {
                    "url": link,
                    "domain": domain,
                    "reason": "swarm_discovered_link",
                    "cached": False,
                    "seeded": False,
                    "discovered_from": document.url,
                }
            )
            if len(candidates) >= limit:
                break
        return candidates

    def _extract_links(self, base_url: str, raw_bytes: bytes) -> List[str]:
        html = raw_bytes[:262144].decode("utf-8", errors="ignore")
        links: List[str] = []
        seen: set[str] = set()
        for match in HREF_RE.finditer(html):
            raw_href = match.group(1).strip()
            if raw_href.startswith(("mailto:", "javascript:", "tel:", "data:")):
                continue
            url = urljoin(base_url, raw_href)
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if self._looks_like_asset(parsed.path):
                continue
            normalized = parsed._replace(fragment="").geturl()
            if normalized in seen:
                continue
            seen.add(normalized)
            links.append(normalized)
            if len(links) >= 128:
                break
        return links

    @staticmethod
    def _looks_like_asset(path: str) -> bool:
        return path.lower().endswith(
            (
                ".7z",
                ".avi",
                ".bmp",
                ".css",
                ".gif",
                ".ico",
                ".jpg",
                ".jpeg",
                ".js",
                ".mov",
                ".mp3",
                ".mp4",
                ".png",
                ".svg",
                ".tar",
                ".webp",
                ".woff",
                ".woff2",
                ".zip",
            )
        )

    def _link_expansion_limit(self) -> int:
        return link_expansion_limit()

    def _extract_title(self, raw_event: RawFetchEvent) -> str:
        prefix = raw_event.raw_bytes[:32768].decode("utf-8", errors="replace")
        match = TITLE_RE.search(prefix) or HEADER_TITLE_RE.search(prefix)
        if not match:
            parsed = urlparse(raw_event.url)
            return parsed.path.rsplit("/", 1)[-1] or parsed.netloc
        title = TAG_RE.sub(" ", match.group(1))
        return re.sub(r"\s+", " ", title).strip() or urlparse(raw_event.url).netloc

    def _split_text_blocks(self, text: str) -> List[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        paragraphs = [re.sub(r"\s+", " ", chunk).strip() for chunk in re.split(r"\n{2,}", normalized)]
        blocks: List[str] = []
        for paragraph in paragraphs:
            if len(paragraph) < MIN_BLOCK_CHARS:
                continue
            if len(paragraph) <= MAX_BLOCK_CHARS:
                blocks.append(paragraph)
                continue
            start = 0
            while start < len(paragraph):
                end = min(start + MAX_BLOCK_CHARS, len(paragraph))
                if end < len(paragraph):
                    split_at = paragraph.rfind(". ", start, end)
                    if split_at > start + MIN_BLOCK_CHARS:
                        end = split_at + 1
                chunk = paragraph[start:end].strip()
                if len(chunk) >= MIN_BLOCK_CHARS:
                    blocks.append(chunk)
                start = end
        if not blocks and normalized.strip():
            blocks.append(normalized.strip()[:MAX_BLOCK_CHARS])
        return blocks[:24]

    def _extract_readable_text(self, raw_bytes: bytes, headers: Dict[str, str]) -> str:
        content_type = str(headers.get("content-type", "")).lower()
        raw_text = raw_bytes[:MAX_RAW_CONTENT_BYTES].decode("utf-8", errors="replace")
        if not raw_text.strip():
            return ""
        if "json" in content_type or raw_text.lstrip().startswith(("{", "[")):
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                return self._compact_plain_text(raw_text)
            strings: List[str] = []
            self._collect_json_strings(payload, strings)
            return self._compact_plain_text("\n".join(strings))
        text = re.sub(r"(?is)<(script|style|noscript|template|svg|canvas)\b.*?</\1>", " ", raw_text)
        text = re.sub(r"(?i)<\s*(br|p|div|li|tr|td|th|h[1-6]|section|article|header|footer)\b[^>]*>", "\n", text)
        text = TAG_RE.sub(" ", text)
        return self._compact_plain_text(html.unescape(text))

    def _collect_json_strings(self, value: Any, strings: List[str], *, depth: int = 0) -> None:
        if depth > 8 or len(strings) >= 512:
            return
        if isinstance(value, str):
            cleaned = self._compact_plain_text(value)
            if len(cleaned) >= 24:
                strings.append(cleaned)
            return
        if isinstance(value, list):
            for item in value[:128]:
                self._collect_json_strings(item, strings, depth=depth + 1)
            return
        if isinstance(value, dict):
            preferred_keys = ("title", "name", "description", "summary", "snippet", "extract", "content", "text")
            for key in preferred_keys:
                if key in value:
                    self._collect_json_strings(value[key], strings, depth=depth + 1)
            for key, item in list(value.items())[:128]:
                if key not in preferred_keys:
                    self._collect_json_strings(item, strings, depth=depth + 1)

    def _compact_plain_text(self, text: str) -> str:
        lines = []
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            compact = re.sub(r"\s+", " ", line).strip()
            if not compact:
                continue
            if self._looks_like_css_or_code_noise(compact):
                continue
            lines.append(compact)
        return "\n\n".join(lines)

    def _select_signal_text(self, *, query: str, candidates: List[str]) -> str:
        usable = [text.strip() for text in candidates if text and text.strip()]
        if not usable:
            return ""
        return max(usable, key=lambda text: (self._text_quality_score(query, text), len(text)))

    def _text_quality_score(self, query: str, text: str) -> float:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return -1000.0
        terms = self._query_terms(query)
        score = min(len(compact) / 500.0, 6.0)
        score += sum(1 for term in terms if term in compact.lower())
        score -= self._noise_penalty(compact)
        if self._definition_cue_score(query, compact) > 0:
            score += 4.0
        return score

    def _rank_documents(self, query: str, documents: List[SearchDocument]) -> List[Dict[str, Any]]:
        terms = self._query_terms(query)
        ranked: List[Dict[str, Any]] = []
        for document in documents:
            for block in document.blocks:
                block_lower = block.lower()
                term_hits = sum(1 for term in terms if term in block_lower)
                title_hits = sum(1 for term in terms if term in document.title.lower())
                if terms and term_hits == 0 and title_hits == 0:
                    continue
                score = float(term_hits * 3 + title_hits * 2)
                score += min(document.classification_confidence, 1.0)
                score += min(len(block) / 600.0, 1.5)
                score += self._definition_cue_score(query, block)
                score += self._subject_definition_score(query, block)
                score -= self._search_page_penalty(document, block)
                score -= self._noise_penalty(block)
                ranked.append(
                    {
                        "url": document.url,
                        "domain": document.domain,
                        "title": document.title,
                        "text": block,
                        "score": round(score, 4),
                        "topology_class": document.topology_class,
                        "classification_confidence": round(document.classification_confidence, 4),
                        "fetch_mode": document.fetch_mode,
                    }
                )
        if not ranked:
            for document in documents[:MAX_SEARCH_BLOCKS]:
                if not document.blocks:
                    continue
                ranked.append(
                    {
                        "url": document.url,
                        "domain": document.domain,
                        "title": document.title,
                        "text": document.blocks[0],
                        "score": round(document.classification_confidence, 4),
                        "topology_class": document.topology_class,
                        "classification_confidence": round(document.classification_confidence, 4),
                        "fetch_mode": document.fetch_mode,
                    }
                )
        ranked.sort(key=lambda item: (-float(item["score"]), item["url"]))
        ranked = self._diversify_ranked_blocks(ranked)
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index
        return ranked

    def _definition_cue_score(self, query: str, text: str) -> float:
        terms = self._query_terms(query)
        if not terms:
            return 0.0
        lowered = re.sub(r"\s+", " ", text.lower())
        score = 0.0
        for term in terms[:4]:
            if re.search(rf"\b{re.escape(term)}\b[^.!?]{{0,120}}\b(is|are|was|were|refers to|means|describes)\b", lowered):
                score += 4.0
            if re.search(rf"\b(is|are|was|were|called|known as)\b[^.!?]{{0,120}}\b{re.escape(term)}\b", lowered):
                score += 1.5
        return min(score, 8.0)

    def _subject_definition_score(self, query: str, text: str) -> float:
        terms = self._query_terms(query)
        if not terms:
            return 0.0
        phrase = r"\s+".join(re.escape(term) for term in terms[:4])
        first = re.escape(terms[0])
        lowered = re.sub(r"\s+", " ", text.lower()).strip(" -:;")
        head = lowered[:260]
        present_copula = r"(?:is|are|refers to|means)"
        past_copula = r"(?:was|were)"
        asks_present = re.match(r"^(what|who|where)\s+(is|are)\b", query.lower().strip()) is not None
        present_subject = rf"{phrase}\b(?:\s*\([^)]{{1,120}}\))?(?:\s+{phrase}\b)?\s+{present_copula}\b"
        past_subject = rf"{phrase}\b(?:\s*\([^)]{{1,120}}\))?(?:\s+{phrase}\b)?\s+{past_copula}\b"
        if re.match(present_subject, head):
            return 11.0 if asks_present else 9.0
        if re.search(rf"\b{present_subject}", head):
            return 7.0 if asks_present else 6.0
        if re.match(past_subject, head):
            return 2.0 if asks_present else 9.0
        if re.search(rf"\b{past_subject}", head):
            return 1.0 if asks_present else 6.0
        if re.match(rf"{first}\s+(?!{present_copula}\b|{past_copula}\b|\(|,|:)", head):
            return -2.0
        return 0.0

    def _search_page_penalty(self, document: SearchDocument, block: str) -> float:
        lowered_title = document.title.lower()
        lowered_url = document.url.lower()
        penalty = 0.0
        if "search result" in lowered_title or lowered_title.startswith("search |"):
            penalty += 2.5
        if any(marker in lowered_url for marker in ("/search", "search?", "site-search", "special:search", "w/index.php?search=")):
            penalty += 1.5
        if "you searched for:" in block.lower():
            penalty -= 1.0
        return penalty

    def _noise_penalty(self, text: str) -> float:
        if not text:
            return 10.0
        sample = text[:2000]
        punctuation = sum(1 for ch in sample if ch in "{}[];:=<>")
        ratio = punctuation / max(1, len(sample))
        penalty = ratio * 18.0
        lowered = sample.lower()
        noisy_markers = ("ve-init-", "mw-parser-output", "function(", "var ", "stylesheet", "<svg", "{\"")
        penalty += sum(1.5 for marker in noisy_markers if marker in lowered)
        if self._looks_like_css_or_code_noise(sample):
            penalty += 4.0
        return penalty

    def _looks_like_css_or_code_noise(self, text: str) -> bool:
        sample = text[:1000].strip()
        if not sample:
            return False
        css_tokens = sample.count("{") + sample.count("}") + sample.count(";")
        if css_tokens >= 8 and css_tokens > len(sample.split()) / 3:
            return True
        if len(re.findall(r"\.[a-zA-Z0-9_-]+\s*\{", sample)) >= 2:
            return True
        return False

    def _build_answer(self, query: str, blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not blocks:
            return None
        best: Optional[tuple[float, Dict[str, Any], str]] = None
        for block in blocks:
            text = str(block.get("text") or "")
            sentence = self._best_answer_sentence(query, text)
            if not sentence:
                sentence = re.sub(r"\s+", " ", text).strip()[:700]
            sentence_score = self._definition_cue_score(query, sentence) + float(block.get("score", 0.0))
            sentence_score -= self._noise_penalty(sentence)
            if best is None or sentence_score > best[0]:
                best = (sentence_score, block, sentence)
        if best is None:
            return None
        _, block, sentence = best
        return {
            "text": sentence,
            "source": {
                "url": block.get("url"),
                "domain": block.get("domain"),
                "title": block.get("title"),
                "score": block.get("score"),
                "rank": block.get("rank"),
            },
        }

    def _best_answer_sentence(self, query: str, text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return ""
        compact = re.sub(r"^.*?\bYou searched for:\s*", "", compact, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        terms = self._query_terms(query)
        best_sentence = ""
        best_score = -1000.0
        for sentence in sentences:
            cleaned = sentence.strip(" -:;")
            if len(cleaned) < 24:
                continue
            lowered = cleaned.lower()
            term_hits = sum(1 for term in terms if term in lowered)
            score = float(term_hits * 2)
            score += self._definition_cue_score(query, cleaned)
            score += self._subject_definition_score(query, cleaned)
            score -= self._noise_penalty(cleaned)
            if 60 <= len(cleaned) <= 420:
                score += 1.0
            if score > best_score:
                best_sentence = cleaned
                best_score = score
        return self._clean_answer_text(best_sentence)[:700]

    def _clean_answer_text(self, text: str) -> str:
        cleaned = re.sub(r"\(\s*/[^)]{1,160}/[^)]{0,80}\)", "", text)
        cleaned = re.sub(r"\(\s*pronunciation[^)]{0,160}\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\(\s+", "(", cleaned)
        cleaned = re.sub(r"\s+\)", ")", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _diversify_ranked_blocks(self, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(ranked) <= MAX_SEARCH_BLOCKS:
            return ranked[:MAX_SEARCH_BLOCKS]
        selected: List[Dict[str, Any]] = []
        selected_ids: set[int] = set()
        domain_counts: Dict[str, int] = {}
        per_domain_soft_cap = 3
        for item in ranked:
            domain = str(item.get("domain") or "")
            if domain_counts.get(domain, 0) >= per_domain_soft_cap:
                continue
            selected.append(item)
            selected_ids.add(id(item))
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if len(selected) >= MAX_SEARCH_BLOCKS:
                return selected
        for item in ranked:
            if id(item) in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= MAX_SEARCH_BLOCKS:
                break
        return selected

    async def _synthesize(self, query: str, blocks: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> str:
        await asyncio.sleep(0)
        if not blocks:
            return f"AXIOM routed '{query}' through {len(candidates)} learned source(s), but extracted 0 ranked blocks."
        source_count = len({block["url"] for block in blocks})
        return f"AXIOM extracted {len(blocks)} ranked block(s) for '{query}' from {source_count} source(s)."


def parse_command(line: str) -> ParsedCommand:
    stripped = line.strip()
    if stripped.lower() == "axiom":
        return ParsedCommand(command="STATUS", payload="")
    if "|" not in stripped:
        if stripped:
            return ParsedCommand(command="SEARCH", payload=stripped)
        raise ValueError("command must use '<search|fetch|learn|status|quit> | <payload>' syntax")
    match = COMMAND_RE.match(line)
    if not match:
        raise ValueError("command must use '<search|fetch|learn|status|quit> | <payload>' syntax")
    return ParsedCommand(command=match.group(1).upper(), payload=match.group(2))


class AxiomInterface:
    def __init__(self, *, store_dir: Path = Path("store"), history_limit: int = 256, runtime: Optional[AxiomRuntimeContext] = None) -> None:
        self.store_dir = store_dir
        self.runtime = runtime or AxiomRuntimeContext(store_dir=store_dir)
        self.orchestrator = QueryOrchestrator(self.runtime)
        self.learned_domains = self.runtime.learned_domains
        self.queued_work = self.runtime.queued_work
        self.cold_start = self.runtime.cold_start
        self.history: Deque[HistoryItem] = collections.deque(maxlen=history_limit)
        self.metrics = InterfaceMetrics()

    async def handle_line(self, line: str) -> InterfaceResponse:
        started = time.perf_counter()
        parsed = parse_command(line)
        req = InterfaceRequest(query_type=parsed.command, payload=parsed.payload, run_id=str(new_run_id()))
        response = await self.handle_request(req)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record(parsed, response, latency_ms)
        return response

    async def handle_request(self, req: InterfaceRequest) -> InterfaceResponse:
        return await self.orchestrator.handle(req)

    async def handle_json(self, payload: Dict[str, Any]) -> InterfaceResponse:
        if "command" in payload:
            query_type = str(payload.get("command", "")).upper()
            req = InterfaceRequest(
                query_type=query_type,
                payload=str(payload.get("payload") or ""),
                run_id=str(payload.get("run_id") or new_run_id()),
            )
            self._store_json_crawl_plan(payload, req)
            started = time.perf_counter()
            response = await self.handle_request(req)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record(ParsedCommand(query_type, req.payload), response, latency_ms)
            return response
        query_type = str(payload.get("query_type", "")).upper()
        req = InterfaceRequest(query_type=query_type, payload=str(payload.get("payload", "")), run_id=str(payload.get("run_id") or new_run_id()))
        self._store_json_crawl_plan(payload, req)
        started = time.perf_counter()
        response = await self.handle_request(req)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record(ParsedCommand(query_type, req.payload), response, latency_ms)
        return response

    def _store_json_crawl_plan(self, payload: Dict[str, Any], req: InterfaceRequest) -> None:
        if req.query_type != "SEARCH":
            return
        if "crawl_plan" in payload:
            plan = normalize_crawl_plan(payload["crawl_plan"], default_query=req.payload)
        elif "swarm_talk" in payload:
            plan = plan_from_generic_talk(payload["swarm_talk"], default_query=req.payload)
        elif payload.get("swarm") is True:
            plan = plan_from_generic_talk(payload, default_query=req.payload)
        else:
            return
        self.runtime.pending_crawl_plans[req.run_id] = plan

    def _record(self, parsed: ParsedCommand, response: InterfaceResponse, latency_ms: float) -> None:
        if response.status == "ok" and response.message == "status":
            response.data.setdefault("metrics", self.metrics.to_dict())
            response.data.setdefault("recent", [item.to_dict() for item in list(self.history)[-10:]])
        self.metrics.record(parsed.command, response.status, latency_ms)
        self.history.append(
            HistoryItem(
                command=parsed.command,
                payload=parsed.payload,
                status=response.status,
                run_id=response.run_id,
                created_unix=int(time.time()),
                latency_ms=latency_ms,
            )
        )

    @staticmethod
    def valid_http_url(raw: str) -> bool:
        parsed = urlparse(raw)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def normalize_domain(raw: str) -> str:
        raw = raw.strip().lower()
        if not raw:
            return ""
        if "://" in raw:
            parsed = urlparse(raw)
            raw = parsed.netloc
        raw = raw.split("/")[0].strip(".")
        if not raw or "." not in raw or any(ch.isspace() for ch in raw):
            return ""
        return raw


class InterfaceTransport(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class JsonLineCodec:
    """Strict JSONL codec for TUI and test clients."""

    @staticmethod
    def encode(response: InterfaceResponse) -> bytes:
        return (json.dumps(response.__dict__, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    @staticmethod
    def decode_line(line: bytes) -> str:
        raw = line.decode("utf-8").strip()
        if not raw:
            raise ValueError("empty command line")
        if raw.startswith("{"):
            obj = json.loads(raw)
            command = obj.get("command")
            payload = obj.get("payload", "")
            if not isinstance(command, str):
                raise ValueError("JSON command requires string field 'command'")
            return f"{command} | {payload}"
        return raw


class InterfaceSocketServer:
    """
    JSONL socket server for the public AXIOM interface.

    Linux production uses Unix sockets. Windows development uses TCP because
    Unix-socket behavior differs across Python/Windows versions.
    """

    def __init__(
        self,
        *,
        interface: Optional[AxiomInterface] = None,
        unix_socket: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 8766,
    ) -> None:
        self.interface = interface or AxiomInterface()
        self.unix_socket = unix_socket or self.interface.runtime.paths.interface_socket
        self.host = host
        self.port = port
        self.server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        use_tcp = os.name == "nt" or self.port != 8766
        if not use_tcp:
            if self.unix_socket.exists():
                self.unix_socket.unlink()
            self.server = await asyncio.start_unix_server(self._handle_client, path=str(self.unix_socket))
        else:
            self.server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if os.name != "nt" and self.unix_socket.exists():
            self.unix_socket.unlink()

    async def serve_forever(self) -> None:
        if self.server is None:
            await self.start()
        assert self.server is not None
        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    command_line = JsonLineCodec.decode_line(line)
                    response = await self.interface.handle_line(command_line)
                except Exception as exc:  # noqa: BLE001 - public boundary returns structured error
                    response = InterfaceResponse(
                        run_id=str(new_run_id()),
                        status="error",
                        message=str(exc),
                        data={"error_type": type(exc).__name__},
                    )
                writer.write(JsonLineCodec.encode(response))
                await writer.drain()
                if response.data.get("quit"):
                    break
        finally:
            writer.close()
            await writer.wait_closed()


def _format_clean_result(response: InterfaceResponse, elapsed_s: float) -> str:
    """Format a beautiful, minimal JSON result from the full response."""
    data = response.data or {}
    query = data.get("query", "")
    blocks = data.get("blocks", [])
    answer = data.get("answer") if isinstance(data.get("answer"), dict) else None
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    if response.status == "error":
        return json.dumps({
            "status": "error",
            "message": response.message,
            "time": ts,
            "time_taken_s": round(elapsed_s, 3),
            "run_id": response.run_id,
        }, indent=2, ensure_ascii=False)

    if answer and isinstance(answer.get("structured"), dict):
        structured = dict(answer.get("structured") or {})
        sources = answer.get("sources") if isinstance(answer.get("sources"), list) else []
        if not sources and answer.get("source"):
            sources = [answer["source"]]
        return json.dumps({
            "status": "ok",
            "query": query,
            "answer": structured,
            "sources": sources[:8],
            "time": ts,
            "time_taken_s": round(elapsed_s, 3),
            "run_id": response.run_id,
        }, indent=2, ensure_ascii=False)

    if answer and answer.get("text"):
        return json.dumps({
            "status": "ok",
            "query": query,
            "answer": answer.get("text"),
            "source": answer.get("source"),
            "time": ts,
            "time_taken_s": round(elapsed_s, 3),
            "run_id": response.run_id,
        }, indent=2, ensure_ascii=False)

    top = blocks[0] if blocks else None
    if top:
        return json.dumps({
            "status": "ok",
            "query": query,
            "answer": top["text"],
            "source": {
                "url": top["url"],
                "domain": top["domain"],
                "title": top["title"],
                "score": top["score"],
            },
            "time": ts,
            "time_taken_s": round(elapsed_s, 3),
            "run_id": response.run_id,
        }, indent=2, ensure_ascii=False)

    # Fallback: status/learn/quit or search with zero results
    return json.dumps({
        "status": response.status,
        "query": query or None,
        "message": response.message,
        "time": ts,
        "time_taken_s": round(elapsed_s, 3),
        "run_id": response.run_id,
    }, indent=2, ensure_ascii=False)


def _capture_preview(buffer: io.StringIO) -> Dict[str, Any]:
    captured = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)
    lines = [line for line in captured.splitlines() if line.strip()]
    return {"captured_lines": len(lines), "preview": lines[:8]}


async def _serve_stdio_loop(
    interface: AxiomInterface,
    loop: asyncio.AbstractEventLoop,
    public_stdout: Any,
    *,
    debug_mode: bool,
    internal_output: Optional[io.StringIO] = None,
) -> int:
    while True:
        line = await loop.run_in_executor(None, os.sys.stdin.readline)
        if not line:
            return 0
        if internal_output is not None:
            stale_capture = _capture_preview(internal_output)
            if stale_capture["captured_lines"]:
                interface.runtime.enqueue({"type": "background_internal_output_captured", **stale_capture})
        wall_start = time.time()
        try:
            response = await interface.handle_line(line)
        except Exception as exc:  # noqa: BLE001 - public interface returns structured error
            response = InterfaceResponse(run_id=str(new_run_id()), status="error", message=str(exc), data={"error_type": type(exc).__name__})
        capture_info = {"captured_lines": 0, "preview": []}
        if internal_output is not None:
            capture_info = _capture_preview(internal_output)
            if capture_info["captured_lines"]:
                interface.runtime.enqueue({"type": "internal_output_captured", "run_id": response.run_id, **capture_info})
        elapsed_s = time.time() - wall_start

        if debug_mode:
            public_stdout.write(json.dumps(response.__dict__, sort_keys=True) + "\n")
        else:
            public_stdout.write(_format_clean_result(response, elapsed_s) + "\n")
        public_stdout.flush()

        if response.data.get("quit"):
            return 0


async def serve_stdio() -> int:
    cfg = load_config()
    start_once_per_login(cfg)
    mark_block("interface.stdio")
    debug_mode = cfg.bool("runtime.debug", False)
    capture_internal = cfg.bool("runtime.capture_internal_output", True)

    interface = AxiomInterface()
    loop = asyncio.get_running_loop()
    public_stdout = os.sys.stdout
    internal_output = io.StringIO()
    if debug_mode or not capture_internal:
        return await _serve_stdio_loop(interface, loop, public_stdout, debug_mode=debug_mode)
    with contextlib.redirect_stdout(internal_output), contextlib.redirect_stderr(internal_output):
        return await _serve_stdio_loop(
            interface,
            loop,
            public_stdout,
            debug_mode=debug_mode,
            internal_output=internal_output,
        )


async def serve_socket() -> int:
    server = InterfaceSocketServer()
    await server.serve_forever()
    return 0


def main() -> int:
    if "--socket" in os.sys.argv:
        return asyncio.run(serve_socket())
    return asyncio.run(serve_stdio())


if __name__ == "__main__":
    raise SystemExit(main())
