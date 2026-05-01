package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestInitializeAndListTools(t *testing.T) {
	server := NewMCPServer(ServerConfig{
		Root:                 ".",
		ProtocolVersion:      defaultProtocolVersion,
		ServerName:           defaultServerName,
		ServerTitle:          "AXIOM TAG MCP Server",
		ServerVersion:        defaultServerVersion,
		Python:               "python",
		WorkerTimeout:        time.Second,
		AnchorTimeout:        time.Second,
		MaxAnchorResults:     3,
		EnableNetworkAnchors: false,
	})
	initID := json.RawMessage(`1`)
	result, rpcErr := server.HandleRequest(nilContext(), JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      &initID,
		Method:  "initialize",
		Params:  json.RawMessage(`{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test"}}`),
	})
	if rpcErr != nil {
		t.Fatalf("initialize rpc error: %#v", rpcErr)
	}
	initResult := result.(map[string]any)
	if initResult["protocolVersion"] != "2025-11-25" {
		t.Fatalf("unexpected protocol version: %#v", initResult["protocolVersion"])
	}
	result, rpcErr = server.HandleRequest(nilContext(), JSONRPCRequest{JSONRPC: "2.0", ID: &initID, Method: "tools/list"})
	if rpcErr != nil {
		t.Fatalf("tools/list rpc error: %#v", rpcErr)
	}
	tools := result.(map[string]any)["tools"].([]ToolDefinition)
	if len(tools) < 9 {
		t.Fatalf("expected MCP tools, got %d", len(tools))
	}
	names := map[string]bool{}
	for _, tool := range tools {
		names[tool.Name] = true
	}
	for _, name := range []string{"tag.search", "tag.expand", "tag.veritas", "tag.inject_context", "anchor.wikipedia", "anchor.news", "anchor.scholar", "anchor.wayback", "anchor.web"} {
		if !names[name] {
			t.Fatalf("missing tool %s", name)
		}
	}
}

func TestSanitizersAndEndpointBuilder(t *testing.T) {
	clean := CompactText(StripTags(`GitHub <b>is</b> &amp; remains a platform.`))
	if clean != "GitHub is & remains a platform." {
		t.Fatalf("unexpected cleaned text: %q", clean)
	}
	if subject := AnchorSubjectQuery("what is a car?"); subject != "car" {
		t.Fatalf("unexpected subject query: %q", subject)
	}
	endpoint := EndpointWithQuery("https://example.test/api?format=json", map[string]string{"query": "what is github", "limit": "3"})
	if !strings.Contains(endpoint, "format=json") || !strings.Contains(endpoint, "query=what+is+github") || !strings.Contains(endpoint, "limit=3") {
		t.Fatalf("bad endpoint: %s", endpoint)
	}
}

func nilContext() contextShim {
	return contextShim{}
}

type contextShim struct{}

func (contextShim) Deadline() (deadline time.Time, ok bool) { return time.Time{}, false }
func (contextShim) Done() <-chan struct{}                   { return nil }
func (contextShim) Err() error                              { return nil }
func (contextShim) Value(key any) any                       { return nil }
