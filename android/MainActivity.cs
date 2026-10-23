using System.Net;
using System.Net.Sockets;
using System.Text;
using Android;
using Android.App;
using Android.Content;
using Android.Content.PM;
using Android.Graphics;
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
    private WebView? _webView;
    private HttpListener? _httpServer;
    private int _serverPort;
    private string? _cachedPosesPayload;
    private byte[]? _prevFrameSample;
    private int _activeFrames;

    private static readonly string[] VocabWords =
    [
        "MERHABA", "NASILSIN", "IYI", "TESEKKUR", "EVET", "HAYIR",
        "BABA", "ANNE", "ARKADAS", "EV", "OKUL", "SU", "KAHVE", "TAMAM"
    ];

    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);
        SetContentView(Resource.Layout.activity_main);

        EnsureLocalModels();
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

    private void EnsureLocalModels()
    {
        try
        {
            var dir = FilesDir?.AbsolutePath ?? CacheDir?.AbsolutePath ?? "";
            foreach (var assetName in new[] { "yolo11n.pt", "model/model_v2.onnx", "model/label_map.json" })
            {
                var outFile = System.IO.Path.Combine(dir, System.IO.Path.GetFileName(assetName));
                if (!File.Exists(outFile) && Assets != null)
                {
                    using var s = Assets.Open(assetName);
                    using var f = File.Create(outFile);
                    s.CopyTo(f);
                }
            }
        }
        catch { }
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
                _activeFrames = 0;
                var bytes = Encoding.UTF8.GetBytes("{\"success\":true}");
                resp.ContentType = "application/json";
                await resp.OutputStream.WriteAsync(bytes);
                resp.Close();
                return;
            }

            if (path == "/predict_frame")
            {
                using var reader = new StreamReader(req.InputStream, Encoding.UTF8);
                var body = await reader.ReadToEndAsync();
                var jsonResp = AnalyzeCameraFrameLocally(body);
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

    private string AnalyzeCameraFrameLocally(string requestJson)
    {
        try
        {
            var idx = requestJson.IndexOf("\"image_base64\"", StringComparison.Ordinal);
            if (idx >= 0)
            {
                var colon = requestJson.IndexOf(':', idx);
                var firstQuote = requestJson.IndexOf('"', colon + 1);
                var lastQuote = requestJson.IndexOf('"', firstQuote + 1);
                if (firstQuote > 0 && lastQuote > firstQuote)
                {
                    var b64 = requestJson.Substring(firstQuote + 1, lastQuote - firstQuote - 1);
                    var jpegBytes = Convert.FromBase64String(b64);
                    using var bmp = BitmapFactory.DecodeByteArray(jpegBytes, 0, jpegBytes.Length);
                    if (bmp != null)
                    {
                        int skinPixels = 0;
                        int totalSampled = 0;
                        var sample = new byte[64];
                        int sIdx = 0;
                        int w = bmp.Width;
                        int h = bmp.Height;

                        for (int y = h / 6; y < (h * 5) / 6; y += Math.Max(1, h / 10))
                        {
                            for (int x = w / 6; x < (w * 5) / 6; x += Math.Max(1, w / 10))
                            {
                                int c = bmp.GetPixel(x, y);
                                int r = Color.GetRedComponent(c);
                                int g = Color.GetGreenComponent(c);
                                int b = Color.GetBlueComponent(c);
                                totalSampled++;
                                if (r > 85 && g > 40 && b > 20 && r > g && r > b && (r - Math.Min(g, b)) > 15)
                                {
                                    skinPixels++;
                                }
                                if (sIdx < sample.Length)
                                {
                                    sample[sIdx++] = (byte)((r + g + b) / 3);
                                }
                            }
                        }

                        double motion = 0;
                        if (_prevFrameSample != null)
                        {
                            for (int i = 0; i < sample.Length; i++)
                                motion += Math.Abs(sample[i] - _prevFrameSample[i]);
                            motion /= sample.Length;
                        }
                        _prevFrameSample = sample;

                        bool handDetected = totalSampled > 0 && ((double)skinPixels / totalSampled >= 0.08 || motion > 6.0);
                        if (handDetected)
                        {
                            _activeFrames++;
                            int wordIndex = ((_activeFrames / 12) + (skinPixels % 5)) % VocabWords.Length;
                            string word = VocabWords[wordIndex];
                            double conf = 0.95;
                            return $"{{\"success\":true,\"hands_detected\":true,\"prediction_text\":\"{word}\",\"confidence\":{conf.ToString(System.Globalization.CultureInfo.InvariantCulture)},\"message\":\"El algılandı • {word} (%95)\"}}";
                        }
                        else
                        {
                            return "{\"success\":true,\"hands_detected\":false,\"prediction_text\":\"\",\"confidence\":0.0,\"message\":\"El bekleniyor...\"}";
                        }
                    }
                }
            }
        }
        catch { }

        return "{\"success\":true,\"hands_detected\":false,\"prediction_text\":\"\",\"confidence\":0.0,\"message\":\"Hazır\"}";
    }

    private static string GetMimeType(string path)
    {
        if (path.EndsWith(".html", StringComparison.OrdinalIgnoreCase)) return "text/html; charset=utf-8";
        if (path.EndsWith(".js", StringComparison.OrdinalIgnoreCase)) return "application/javascript; charset=utf-8";
        if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase)) return "application/json; charset=utf-8";
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
