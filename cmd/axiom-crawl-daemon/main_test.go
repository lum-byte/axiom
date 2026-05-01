package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestResidentWorkersStayStartedAcrossQueries(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/a", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/html")
		_, _ = w.Write([]byte(`<html><head><title>Alpha</title></head><body>GitHub stores and manages code.<a href="/b">Beta</a></body></html>`))
	})
	mux.HandleFunc("/b", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/html")
		_, _ = w.Write([]byte(`<html><head><title>Beta</title></head><body>GitHub provides distributed version control.</body></html>`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	daemon := NewDaemon(Config{Workers: 3, RequestTimeoutMS: 2000})
	defer daemon.Shutdown()
	if daemon.workersStarted.Load() != 3 {
		t.Fatalf("workers did not boot once: %d", daemon.workersStarted.Load())
	}
	req := Request{
		ID:    "q1",
		Op:    "query",
		Query: "what is github",
		Candidates: []Candidate{
			{URL: server.URL + "/a", Domain: "example.test"},
			{URL: server.URL + "/b", Domain: "example.test"},
		},
		Limit: 2,
	}
	resp := daemon.Handle(context.Background(), req)
	if resp.Status != "ok" {
		t.Fatalf("query failed: %#v", resp)
	}
	resp2 := daemon.Handle(context.Background(), req)
	if resp2.Status != "ok" {
		t.Fatalf("second query failed: %#v", resp2)
	}
	if daemon.workersStarted.Load() != 3 {
		t.Fatalf("workers should stay resident, got %d", daemon.workersStarted.Load())
	}
	if daemon.queriesHandled.Load() != 2 {
		t.Fatalf("expected two handled queries, got %d", daemon.queriesHandled.Load())
	}
}

func TestPageRankPromotesReferencedPages(t *testing.T) {
	ranker := NewPageRanker(0.85, 12)
	ranker.Observe("https://a.example/", []string{"https://b.example/"})
	ranker.Observe("https://c.example/", []string{"https://b.example/"})
	if ranker.Score("https://b.example/") <= ranker.Score("https://a.example/") {
		t.Fatalf("expected inbound-linked page to outrank source: b=%f a=%f", ranker.Score("https://b.example/"), ranker.Score("https://a.example/"))
	}
}

func TestJSONLProtocolStatusAndShutdown(t *testing.T) {
	daemon := NewDaemon(Config{Workers: 2, RequestTimeoutMS: 500})
	input := strings.NewReader(`{"id":"s1","op":"status"}` + "\n" + `{"id":"x","op":"shutdown"}` + "\n")
	var output bytes.Buffer
	err := RunJSONL(context.Background(), daemon, input, &output)
	if err != nil {
		t.Fatalf("jsonl failed: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected two responses, got %d: %q", len(lines), output.String())
	}
	var status Response
	if err := json.Unmarshal([]byte(lines[0]), &status); err != nil {
		t.Fatalf("bad status json: %v", err)
	}
	if status.Status != "ok" || status.Data["workers"].(float64) != 2 {
		t.Fatalf("unexpected status: %#v", status)
	}
}

func TestQueryTimeoutReturnsPartialTelemetry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		_, _ = w.Write([]byte(`<html><title>slow</title><body>slow page</body></html>`))
	}))
	defer server.Close()
	daemon := NewDaemon(Config{Workers: 1, RequestTimeoutMS: 50})
	defer daemon.Shutdown()
	resp := daemon.Handle(context.Background(), Request{
		ID:         "slow",
		Op:         "query",
		Query:      "slow",
		Limit:      1,
		TimeoutMS:  25,
		Candidates: []Candidate{{URL: server.URL}},
	})
	if resp.Status != "ok" {
		t.Fatalf("timeout should return ok with telemetry, got %#v", resp)
	}
	telemetry := resp.Data["telemetry"].(map[string]any)
	if telemetry["timed_out"] != true {
		t.Fatalf("expected timed_out telemetry, got %#v", telemetry)
	}
}
