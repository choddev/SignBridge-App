using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using Android;
using Android.App;
using Android.Content.PM;
using Android.OS;
using Android.Webkit;

namespace SignBridgeAndroid;

[Activity(
    Label = "@string/app_name",
    MainLauncher = true,
    HardwareAccelerated = true,
    ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation | ConfigChanges.KeyboardHidden)]
public class MainActivity : Activity
{
    private const int PermRequestCode = 1001;
    private const int SeqLen = 30;
    private const int FaceEnd = 1404;
    private const int PoseEnd = 1503;
    private const int LeftEnd = 1566;
    private const int RightEnd = 1629;
    private const int HandVel = 126;
    private const int InputSize = 1755;
    private const int HiddenSize = 256;
    private const int GateSize = 768; // 3 * 256
    private const int Linear1Size = 128;
    private const int NumClasses = 122;
    private const int ProbAvgWindow = 3;
    private const double InterSignPauseSec = 0.50;

    private WebView? _webView;
    private HttpListener? _httpServer;
    private int _serverPort;
    private string? _cachedPosesPayload;

    // Real V2 GRU Model Weights loaded from Assets/model/model_v2.bin + label_map.json
    private float[]? _modelWeights;
    private string[] _idxToLabel = new string[NumClasses];
    private Dictionary<string, int> _labelToIdx = new(StringComparer.OrdinalIgnoreCase);
    private bool _modelLoaded;
    private readonly object _inferLock = new();

    // Sequence buffer & stabilization state
    private readonly List<float[]> _lmBuffer = new(SeqLen);
    private readonly List<float[]> _giBaseBuffer = new(SeqLen);
    private readonly Queue<float[]> _probHistory = new(ProbAvgWindow);
    private DateTime _lastHandTime = DateTime.MinValue;
    private bool _pauseResetDone;

    private static readonly HashSet<string> DemoWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "MERHABA", "NASILSIN", "IYI", "SEN", "NE", "YAPMAK",
        "BEN", "CALISMAK", "COK", "YORULMAK",
        "SAAT", "KAC", "BITMEK",
        "AKSAM", "BES", "SONRA", "BOS",
        "BERABER", "KAHVE", "ICMEK",
        "TAMAM", "NEREDE", "BULUSMAK",
        "PARK", "YAN", "KAFE",
        "GORUSMEK", "KENDI", "BAKMAK"
    };

    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);
        SetContentView(Resource.Layout.activity_main);

        Task.Run(LoadRealGruModel);
        StartLocalAssetServer();

        _webView = FindViewById<WebView>(Resource.Id.signBridgeWebView);
        if (_webView != null)
        {
            var settings = _webView.Settings;
            settings.JavaScriptEnabled = true;
            settings.DomStorageEnabled = true;
            settings.DatabaseEnabled = true;
            settings.MediaPlaybackRequiresUserGesture = false;
            settings.AllowFileAccess = true;
            settings.AllowContentAccess = true;
            settings.MixedContentMode = MixedContentHandling.AlwaysAllow;

            _webView.SetWebChromeClient(new CameraPermissionChromeClient(this));
            _webView.SetWebViewClient(new WebViewClient());
        }

        if (HasCameraPermission())
        {
            LoadMobileApp();
        }
        else
        {
            RequestPermissions(
                [Manifest.Permission.Camera, Manifest.Permission.RecordAudio],
                PermRequestCode);
            LoadMobileApp();
        }
    }

    private bool HasCameraPermission()
    {
        if (Build.VERSION.SdkInt < BuildVersionCodes.M) return true;
        return CheckSelfPermission(Manifest.Permission.Camera) == Permission.Granted;
    }

    public override void OnRequestPermissionsResult(int requestCode, string[] permissions, Permission[] grantResults)
    {
        base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PermRequestCode && _webView != null)
        {
            RunOnUiThread(() =>
            {
                _webView.EvaluateJavascript("if(typeof startCamera==='function') startCamera();", null);
            });
        }
    }

    private void LoadMobileApp()
    {
        if (_webView == null) return;
        _webView.LoadUrl($"http://127.0.0.1:{_serverPort}/mobile.html");
    }

    private static string NormalizeGlossKey(string k)
    {
        return k.Replace("İ", "I").Replace("ı", "I")
                .Replace("Ş", "S").Replace("ş", "S")
                .Replace("Ç", "C").Replace("ç", "C")
                .Replace("Ğ", "G").Replace("ğ", "G")
                .Replace("Ü", "U").Replace("ü", "U")
                .Replace("Ö", "O").Replace("ö", "O")
                .ToUpperInvariant();
    }

    private void LoadRealGruModel()
    {
        try
        {
            if (Assets == null) return;

            using (var sr = new StreamReader(Assets.Open("model/label_map.json"), Encoding.UTF8))
            {
                var json = sr.ReadToEnd();
                var rawMap = JsonSerializer.Deserialize<Dictionary<string, int>>(json);
                if (rawMap != null)
                {
                    foreach (var kv in rawMap)
                    {
                        var normKey = NormalizeGlossKey(kv.Key);
                        if (kv.Value >= 0 && kv.Value < NumClasses)
                        {
                            _idxToLabel[kv.Value] = normKey;
                            _labelToIdx[normKey] = kv.Value;
                        }
                    }
                }
            }

            using (var s = Assets.Open("model/model_v2.bin"))
            using (var ms = new MemoryStream())
            {
                s.CopyTo(ms);
                var bytes = ms.ToArray();
                int floatCount = bytes.Length / 4;
                var weights = new float[floatCount];
                System.Buffer.BlockCopy(bytes, 0, weights, 0, bytes.Length);
                _modelWeights = weights;
                _modelLoaded = floatCount == 1989370;
            }
        }
        catch
        {
            _modelLoaded = false;
        }
    }

    private void StartLocalAssetServer()
    {
        try
        {
            var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            _serverPort = ((IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();

            _httpServer = new HttpListener();
            _httpServer.Prefixes.Add($"http://127.0.0.1:{_serverPort}/");
            _httpServer.Start();

            Task.Run(async () =>
            {
                while (_httpServer != null && _httpServer.IsListening)
                {
                    try
                    {
                        var ctx = await _httpServer.GetContextAsync();
                        _ = Task.Run(() => HandleHttpRequest(ctx));
                    }
                    catch { break; }
                }
            });
        }
        catch { }
    }

    private async Task HandleHttpRequest(HttpListenerContext ctx)
    {
        var req = ctx.Request;
        var resp = ctx.Response;
        try
        {
            resp.Headers.Add("Access-Control-Allow-Origin", "*");
            resp.Headers.Add("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            resp.Headers.Add("Access-Control-Allow-Headers", "Content-Type");

            if (req.HttpMethod == "OPTIONS")
            {
                resp.StatusCode = 200;
                resp.Close();
                return;
            }

            var path = req.Url?.AbsolutePath ?? "/";
            if (path == "/") path = "/mobile.html";

            if (path == "/api/list-poses")
            {
                if (_cachedPosesPayload == null && Assets != null)
                {
                    using var sr = new StreamReader(Assets.Open("saved_poses.json"), Encoding.UTF8);
                    var rawPoses = await sr.ReadToEndAsync();
                    _cachedPosesPayload = "{\"poses\":" + rawPoses + "}";
                }
                var bytes = Encoding.UTF8.GetBytes(_cachedPosesPayload ?? "{\"poses\":{}}");
                resp.ContentType = "application/json; charset=utf-8";
                resp.ContentLength64 = bytes.Length;
                await resp.OutputStream.WriteAsync(bytes);
                resp.Close();
                return;
            }

            if (path == "/reset_buffer")
            {
                lock (_inferLock)
                {
                    _lmBuffer.Clear();
                    _giBaseBuffer.Clear();
                    _probHistory.Clear();
                }
                var bytes = Encoding.UTF8.GetBytes("{\"success\":true}");
                resp.ContentType = "application/json";
                await resp.OutputStream.WriteAsync(bytes);
                resp.Close();
                return;
            }

            if (path == "/predict_landmarks")
            {
                using var reader = new StreamReader(req.InputStream, Encoding.UTF8);
                var body = await reader.ReadToEndAsync();
                var jsonResp = PredictFromLandmarksJson(body);
                var bytes = Encoding.UTF8.GetBytes(jsonResp);
                resp.ContentType = "application/json; charset=utf-8";
                await resp.OutputStream.WriteAsync(bytes);
                resp.Close();
                return;
            }

            var assetPath = path.TrimStart('/');
            if (Assets != null)
            {
                using var assetStream = Assets.Open(assetPath);
                using var ms = new MemoryStream();
                await assetStream.CopyToAsync(ms);
                var data = ms.ToArray();
                resp.ContentType = GetMimeType(assetPath);
                resp.ContentLength64 = data.Length;
                await resp.OutputStream.WriteAsync(data);
                resp.Close();
                return;
            }
        }
        catch
        {
            try
            {
                resp.StatusCode = 404;
                resp.Close();
            }
            catch { }
        }
    }

    private string PredictFromLandmarksJson(string requestJson)
    {
        try
        {
            if (!_modelLoaded || _modelWeights == null)
            {
                return "{\"success\":true,\"hands_detected\":false,\"prediction_text\":\"\",\"confidence\":0.0,\"buffer_size\":0,\"message\":\"Model yükleniyor...\"}";
            }

            using var doc = JsonDocument.Parse(requestJson);
            var root = doc.RootElement;
            bool handVisible = root.TryGetProperty("hand_visible", out var hvProp) && hvProp.GetBoolean();
            bool demoMode = root.TryGetProperty("demo_mode", out var dmProp) && dmProp.GetBoolean();

            var now = DateTime.UtcNow;

            lock (_inferLock)
            {
                if (!handVisible)
                {
                    if (!_pauseResetDone && _lastHandTime != DateTime.MinValue &&
                        (now - _lastHandTime).TotalSeconds >= InterSignPauseSec)
                    {
                        _lmBuffer.Clear();
                        _giBaseBuffer.Clear();
                        _probHistory.Clear();
                        _pauseResetDone = true;
                    }
                    return $"{{\"success\":true,\"hands_detected\":false,\"prediction_text\":\"\",\"confidence\":0.0,\"buffer_size\":{_lmBuffer.Count},\"message\":\"Ellerinizi gösterin...\"}}";
                }

                if (!root.TryGetProperty("landmarks", out var lmProp) || lmProp.ValueKind != JsonValueKind.Array)
                {
                    return "{\"success\":false,\"error\":\"Missing landmarks\"}";
                }

                var lm = new float[RightEnd];
                int idx = 0;
                foreach (var el in lmProp.EnumerateArray())
                {
                    if (idx >= RightEnd) break;
                    lm[idx++] = el.GetSingle();
                }
                if (idx != RightEnd)
                {
                    return "{\"success\":false,\"error\":\"Invalid landmarks length\"}";
                }

                _lastHandTime = now;
                _pauseResetDone = false;

                NormalizeLandmarksInPlace(lm);
                var giBase = ComputeGiBase(lm, _modelWeights);

                if (_lmBuffer.Count >= SeqLen)
                {
                    _lmBuffer.RemoveAt(0);
                    _giBaseBuffer.RemoveAt(0);
                }
                _lmBuffer.Add(lm);
                _giBaseBuffer.Add(giBase);

                int bufLen = _lmBuffer.Count;
                if (bufLen < SeqLen)
                {
                    return $"{{\"success\":true,\"hands_detected\":true,\"prediction_text\":\"\",\"confidence\":0.0,\"buffer_size\":{bufLen},\"message\":\"İşaret okunuyor ({bufLen}/{SeqLen})\"}}";
                }

                var rawProbs = RunDecomposedGru(_lmBuffer, _giBaseBuffer, _modelWeights);
                if (_probHistory.Count >= ProbAvgWindow) _probHistory.Dequeue();
                _probHistory.Enqueue(rawProbs);

                var avgProbs = new float[NumClasses];
                foreach (var p in _probHistory)
                {
                    for (int i = 0; i < NumClasses; i++) avgProbs[i] += p[i];
                }
                float invCount = 1.0f / _probHistory.Count;
                for (int i = 0; i < NumClasses; i++) avgProbs[i] *= invCount;

                string predWord = "";
                double predConf = 0.0;
                double margin = 0.0;

                if (demoMode)
                {
                    double demoSum = 0.0;
                    int bestIdx = -1;
                    double bestP = -1.0;
                    double secondP = -1.0;
                    foreach (var w in DemoWords)
                    {
                        if (_labelToIdx.TryGetValue(w, out int cIdx))
                        {
                            double p = avgProbs[cIdx];
                            demoSum += p;
                            if (p > bestP)
                            {
                                secondP = bestP;
                                bestP = p;
                                bestIdx = cIdx;
                            }
                            else if (p > secondP)
                            {
                                secondP = p;
                            }
                        }
                    }
                    if (bestIdx >= 0 && demoSum >= 0.35)
                    {
                        predWord = _idxToLabel[bestIdx];
                        predConf = (bestP / Math.Max(demoSum, 1e-6)) * Math.Min(1.0, demoSum + 0.35);
                        margin = bestP - Math.Max(0.0, secondP);
                    }
                }
                else
                {
                    int best1 = 0, best2 = 1;
                    if (avgProbs[1] > avgProbs[0]) { best1 = 1; best2 = 0; }
                    for (int i = 2; i < NumClasses; i++)
                    {
                        if (avgProbs[i] > avgProbs[best1])
                        {
                            best2 = best1;
                            best1 = i;
                        }
                        else if (avgProbs[i] > avgProbs[best2])
                        {
                            best2 = i;
                        }
                    }
                    predWord = _idxToLabel[best1];
                    predConf = avgProbs[best1];
                    margin = avgProbs[best1] - avgProbs[best2];
                }

                if (predConf < 0.72 || margin < 0.18)
                {
                    predWord = "";
                    predConf = 0.0;
                }

                string msg = string.IsNullOrEmpty(predWord)
                    ? "El algılandı"
                    : $"El algılandı • {predWord} (%{(int)Math.Round(predConf * 100)})";

                return $"{{\"success\":true,\"hands_detected\":true,\"prediction_text\":\"{predWord}\",\"confidence\":{predConf.ToString(CultureInfo.InvariantCulture)},\"margin\":{margin.ToString(CultureInfo.InvariantCulture)},\"buffer_size\":{bufLen},\"message\":\"{msg}\"}}";
            }
        }
        catch (Exception ex)
        {
            return $"{{\"success\":false,\"error\":\"{ex.Message.Replace("\"", "'")}\"}}";
        }
    }

    private static bool AllZero3(float a, float b, float c)
    {
        const float eps = 1e-7f;
        return MathF.Abs(a) < eps && MathF.Abs(b) < eps && MathF.Abs(c) < eps;
    }

    private static void NormalizeLandmarksInPlace(float[] lm)
    {
        // Layer 1: Shoulder-centered
        int lsOff = FaceEnd + 11 * 3;
        int rsOff = FaceEnd + 12 * 3;
        float ls0 = lm[lsOff], ls1 = lm[lsOff + 1], ls2 = lm[lsOff + 2];
        float rs0 = lm[rsOff], rs1 = lm[rsOff + 1], rs2 = lm[rsOff + 2];

        bool lsZero = AllZero3(ls0, ls1, ls2);
        bool rsZero = AllZero3(rs0, rs1, rs2);

        if (!(lsZero && rsZero))
        {
            float cx, cy, cz;
            if (lsZero) { cx = rs0; cy = rs1; cz = rs2; }
            else if (rsZero) { cx = ls0; cy = ls1; cz = ls2; }
            else { cx = (ls0 + rs0) * 0.5f; cy = (ls1 + rs1) * 0.5f; cz = (ls2 + rs2) * 0.5f; }

            float dx = rs0 - ls0, dy = rs1 - ls1, dz = rs2 - ls2;
            float scale = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
            if (scale >= 1e-4f)
            {
                float inv = 1.0f / scale;
                for (int i = 0; i < RightEnd; i += 3)
                {
                    lm[i] = (lm[i] - cx) * inv;
                    lm[i + 1] = (lm[i + 1] - cy) * inv;
                    lm[i + 2] = (lm[i + 2] - cz) * inv;
                }
            }
        }

        // Layer 2: Wrist-centered per hand
        NormalizeHandInPlace(lm, PoseEnd);
        NormalizeHandInPlace(lm, LeftEnd);
    }

    private static void NormalizeHandInPlace(float[] lm, int handStart)
    {
        float wx = lm[handStart], wy = lm[handStart + 1], wz = lm[handStart + 2];
        if (AllZero3(wx, wy, wz)) return;

        int mOff = handStart + 9 * 3;
        float dx = lm[mOff] - wx, dy = lm[mOff + 1] - wy, dz = lm[mOff + 2] - wz;
        float palmScale = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
        if (palmScale <= 1e-4f) return;

        float inv = 1.0f / palmScale;
        for (int p = 1; p < 21; p++)
        {
            int i = handStart + p * 3;
            lm[i] = (lm[i] - wx) * inv;
            lm[i + 1] = (lm[i + 1] - wy) * inv;
            lm[i + 2] = (lm[i + 2] - wz) * inv;
        }
    }

    private static float Clamp5(float v) => v < -5.0f ? -5.0f : (v > 5.0f ? 5.0f : v);

    private static float Sigmoid(float x)
    {
        if (x < -50.0f) x = -50.0f;
        else if (x > 50.0f) x = 50.0f;
        return 1.0f / (1.0f + MathF.Exp(-x));
    }

    private static float[] ComputeGiBase(float[] lm, float[] w)
    {
        // w_ih_l0: offset 0, shape (768, 1755)
        // b_ih_l0: offset 1937664, shape (768)
        const int bIhL0Off = 1937664;
        var giBase = new float[GateSize];
        var lmClipped = new float[RightEnd];
        for (int j = 0; j < RightEnd; j++) lmClipped[j] = Clamp5(lm[j]);

        for (int r = 0; r < GateSize; r++)
        {
            int rowOff = r * InputSize;
            float sum = w[bIhL0Off + r];
            for (int j = 0; j < RightEnd; j++)
            {
                sum += w[rowOff + j] * lmClipped[j];
            }
            giBase[r] = sum;
        }
        return giBase;
    }

    private static float[] RunDecomposedGru(List<float[]> lmBuf, List<float[]> giBaseBuf, float[] w)
    {
        const int wHhL0Off = 1347840; // (768, 256)
        const int wIhL1Off = 1544448; // (768, 256)
        const int wHhL1Off = 1741056; // (768, 256)
        const int bHhL0Off = 1938432; // (768)
        const int bIhL1Off = 1939200; // (768)
        const int bHhL1Off = 1939968; // (768)
        const int w1Off = 1940736;    // (128, 256)
        const int b1Off = 1973504;    // (128)
        const int w2Off = 1973632;    // (122, 128)
        const int b2Off = 1989248;    // (122)

        var h0 = new float[HiddenSize];
        var h1 = new float[HiddenSize];
        var h0Next = new float[HiddenSize];
        var h1Next = new float[HiddenSize];
        var gi0 = new float[GateSize];
        var gh0 = new float[GateSize];
        var gi1 = new float[GateSize];
        var gh1 = new float[GateSize];
        var vel = new float[HandVel];

        for (int t = 0; t < SeqLen; t++)
        {
            var baseT = giBaseBuf[t];
            if (t == 0)
            {
                Array.Copy(baseT, gi0, GateSize);
            }
            else
            {
                var curLm = lmBuf[t];
                var prevLm = lmBuf[t - 1];
                for (int j = 0; j < HandVel; j++)
                {
                    vel[j] = Clamp5(curLm[PoseEnd + j] - prevLm[PoseEnd + j]);
                }
                for (int r = 0; r < GateSize; r++)
                {
                    int rowOff = r * InputSize + RightEnd;
                    float sum = baseT[r];
                    for (int j = 0; j < HandVel; j++)
                    {
                        sum += w[rowOff + j] * vel[j];
                    }
                    gi0[r] = sum;
                }
            }

            // Layer 0 recurrent: gh0 = h0 @ w_hh_l0.T + b_hh_l0
            for (int r = 0; r < GateSize; r++)
            {
                int rowOff = wHhL0Off + r * HiddenSize;
                float sum = w[bHhL0Off + r];
                for (int j = 0; j < HiddenSize; j++)
                {
                    sum += w[rowOff + j] * h0[j];
                }
                gh0[r] = sum;
            }

            for (int i = 0; i < HiddenSize; i++)
            {
                float rGate = Sigmoid(gi0[i] + gh0[i]);
                float zGate = Sigmoid(gi0[HiddenSize + i] + gh0[HiddenSize + i]);
                float nGate = MathF.Tanh(gi0[512 + i] + rGate * gh0[512 + i]);
                h0Next[i] = (1.0f - zGate) * nGate + zGate * h0[i];
            }
            Array.Copy(h0Next, h0, HiddenSize);

            // Layer 1: gi1 = h0 @ w_ih_l1.T + b_ih_l1, gh1 = h1 @ w_hh_l1.T + b_hh_l1
            for (int r = 0; r < GateSize; r++)
            {
                int rowIh = wIhL1Off + r * HiddenSize;
                int rowHh = wHhL1Off + r * HiddenSize;
                float sumIh = w[bIhL1Off + r];
                float sumHh = w[bHhL1Off + r];
                for (int j = 0; j < HiddenSize; j++)
                {
                    sumIh += w[rowIh + j] * h0[j];
                    sumHh += w[rowHh + j] * h1[j];
                }
                gi1[r] = sumIh;
                gh1[r] = sumHh;
            }

            for (int i = 0; i < HiddenSize; i++)
            {
                float rGate = Sigmoid(gi1[i] + gh1[i]);
                float zGate = Sigmoid(gi1[HiddenSize + i] + gh1[HiddenSize + i]);
                float nGate = MathF.Tanh(gi1[512 + i] + rGate * gh1[512 + i]);
                h1Next[i] = (1.0f - zGate) * nGate + zGate * h1[i];
            }
            Array.Copy(h1Next, h1, HiddenSize);
        }

        // Classifier: ReLU(h1 @ w1.T + b1) @ w2.T + b2
        var hidden128 = new float[Linear1Size];
        for (int r = 0; r < Linear1Size; r++)
        {
            int rowOff = w1Off + r * HiddenSize;
            float sum = w[b1Off + r];
            for (int j = 0; j < HiddenSize; j++) sum += w[rowOff + j] * h1[j];
            hidden128[r] = sum > 0.0f ? sum : 0.0f;
        }

        var logits = new float[NumClasses];
        float maxLogit = float.NegativeInfinity;
        for (int c = 0; c < NumClasses; c++)
        {
            int rowOff = w2Off + c * Linear1Size;
            float sum = w[b2Off + c];
            for (int j = 0; j < Linear1Size; j++) sum += w[rowOff + j] * hidden128[j];
            logits[c] = sum;
            if (sum > maxLogit) maxLogit = sum;
        }

        var probs = new float[NumClasses];
        float expSum = 0.0f;
        for (int c = 0; c < NumClasses; c++)
        {
            float e = MathF.Exp(logits[c] - maxLogit);
            probs[c] = e;
            expSum += e;
        }
        float invExpSum = 1.0f / expSum;
        for (int c = 0; c < NumClasses; c++) probs[c] *= invExpSum;

        return probs;
    }

    private static string GetMimeType(string path)
    {
        if (path.EndsWith(".html", StringComparison.OrdinalIgnoreCase)) return "text/html; charset=utf-8";
        if (path.EndsWith(".js", StringComparison.OrdinalIgnoreCase)) return "application/javascript; charset=utf-8";
        if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase)) return "application/json; charset=utf-8";
        if (path.EndsWith(".wasm", StringComparison.OrdinalIgnoreCase)) return "application/wasm";
        if (path.EndsWith(".glb", StringComparison.OrdinalIgnoreCase)) return "model/gltf-binary";
        if (path.EndsWith(".css", StringComparison.OrdinalIgnoreCase)) return "text/css; charset=utf-8";
        if (path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)) return "application/octet-stream";
        return "application/octet-stream";
    }

    private sealed class CameraPermissionChromeClient : WebChromeClient
    {
        private readonly Activity _activity;
        public CameraPermissionChromeClient(Activity activity) => _activity = activity;

        public override void OnPermissionRequest(PermissionRequest? request)
        {
            if (request == null) return;
            _activity.RunOnUiThread(() =>
            {
                try { request.Grant(request.GetResources()); } catch { }
            });
        }
    }
}
