package server

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

type pointerSystemOneTestRunner struct {
	systemOneTestRunner
	port int
}

func (r *pointerSystemOneTestRunner) GetPort() int { return r.port }

func TestPointerSystemOneNativeForwarding(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	// Native validation errors must not fall through to the external runner.
	t.Setenv("OLLAMA_POINTER_RUNNER", "http://127.0.0.1:1")
	_, digest := createBinFile(t, gguftest.KV{
		"general.architecture": "edlm",
		"edlm.context_length":  uint32(32768),
	}, nil)
	configLayer, err := createConfigLayer(model.ConfigV2{
		ModelFormat: "gguf", ModelFamily: "edlm", Capabilities: []string{"decision"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(model.ParseName("pointer-test"), *configLayer, []manifest.Layer{
		{MediaType: "application/vnd.ollama.image.model", Digest: digest},
	}); err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		name     string
		body     string
		status   int
		response string
	}{
		{
			name:     "ordered legend array",
			body:     "{\n  \"model\":\"pointer-test\", \"state\":{\"z\":1,\"a\":2}, \"questions\":{\"rating\":{\"type\":\"score\",\"instructions\":\"Rate\",\"criteria\":[\"low\",\"high\"]}}\n}",
			status:   http.StatusOK,
			response: `{"answers":{"rating":{"type":"score","score":0.75,"legend":["low","high"],"probabilities":{"0":0.25,"1":0.75},"confidence":0.75}},"usage":{"input_tokens":12,"output_tokens":2}}`,
		},
		{
			name:     "batch forwarded unchanged",
			body:     `{"model":"pointer-test","requests":[{"state":"first","questions":{"q":{"type":"noul"}}},{"state":"second","questions":{"q":{"type":"noul"}}}]}`,
			status:   http.StatusOK,
			response: `{"model":"pointer-test","results":[{"answers":{"q":{"type":"noul","noul":0.25}}},{"answers":{"q":{"type":"noul","noul":0.75}}}]}`,
		},
		{
			name:     "empty requests wrapper",
			body:     `{"model":"pointer-test","state":"x","questions":{"q":{"type":"noul"}},"requests":[]}`,
			status:   http.StatusBadRequest,
			response: `{"error":{"code":400,"message":"requests batch wrapper is not supported; send one state and questions object","type":"invalid_request_error"}}`,
		},
		{
			name:     "null requests wrapper",
			body:     `{"model":"pointer-test","state":"x","questions":{"q":{"type":"noul"}},"requests":null}`,
			status:   http.StatusBadRequest,
			response: `{"error":{"code":400,"message":"requests batch wrapper is not supported; send one state and questions object","type":"invalid_request_error"}}`,
		},
		{
			name:     "invalid noul criteria",
			body:     `{"model":"pointer-test","state":"x","questions":{"q":{"type":"noul","criteria":[]}}}`,
			status:   http.StatusBadRequest,
			response: `{"error":{"code":400,"message":"noul criteria must be an object or null","type":"invalid_request_error"}}`,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			received := make(chan string, 1)
			native := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				received <- string(body)
				if r.Method != http.MethodPost || r.URL.Path != "/v1/systemone" {
					t.Errorf("native request: %s %s", r.Method, r.URL.Path)
				}
				if r.Header.Get("x-typesafe-request-id") != "incoming-id" {
					t.Error("native request lost the request ID")
				}
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("x-typesafe-request-id", "native-id")
				w.WriteHeader(tt.status)
				io.WriteString(w, tt.response)
			}))
			defer native.Close()
			runner := &pointerSystemOneTestRunner{port: native.Listener.Addr().(*net.TCPAddr).Port}
			ref := &runnerRef{llama: runner, refCount: 1, sessionDuration: time.Hour}
			s := newServerWithMockRunner(t, &runner.mockRunner)
			s.sched.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				runner.contextLength = req.opts.NumCtx
				ref.model = req.model
				ref.modelKey = schedulerModelKey(req.model)
				s.sched.loadedMu.Lock()
				s.sched.loaded[ref.modelKey] = ref
				s.sched.loadedMu.Unlock()
				req.successCh <- ref
				return false
			}
			router := gin.New()
			router.POST("/v1/systemone", s.SystemOneHandler)
			w := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("x-typesafe-request-id", "incoming-id")
			router.ServeHTTP(w, req)

			select {
			case body := <-received:
				if body != tt.body {
					t.Errorf("native request body changed: %q, want %q", body, tt.body)
				}
			default:
				t.Fatalf("native runner was not called: status=%d body=%s", w.Code, w.Body)
			}
			if w.Code != tt.status || w.Body.String() != tt.response {
				t.Errorf("response changed: status=%d body=%s; want status=%d body=%s", w.Code, w.Body, tt.status, tt.response)
			}
			if w.Header().Get("Content-Type") != "application/json" || w.Header().Get("x-typesafe-request-id") != "native-id" {
				t.Errorf("native response headers changed: %v", w.Header())
			}
			if runner.calls != 0 {
				t.Fatal("pointer-head request was letter-scored")
			}
		})
	}
}
