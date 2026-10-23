using Android.App;
using Android.Content;
using Android.OS;
using Android.Webkit;
using Java.Interop;

namespace SignBridgeAndroid;

[Activity(Label = "@string/app_name", MainLauncher = true, ConfigurationChanges = Android.Content.PM.ConfigChanges.ScreenSize | Android.Content.PM.ConfigChanges.Orientation)]
public class MainActivity : Activity
{
    private WebView? _webView;
    private string _localModelPath = string.Empty;

    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);
        SetContentView(Resource.Layout.activity_main);

        // Extract bundled yolo11n.pt from APK assets to local device storage for local inference
        _localModelPath = EnsureLocalYoloModel();

        _webView = FindViewById<WebView>(Resource.Id.signBridgeWebView);
        if (_webView != null)
        {
            var settings = _webView.Settings;
            settings.JavaScriptEnabled = true;
            settings.DomStorageEnabled = true;
            settings.MediaPlaybackRequiresUserGesture = false;
            settings.AllowFileAccess = true;
            settings.AllowContentAccess = true;

            _webView.SetWebChromeClient(new SignBridgeChromeClient());
            _webView.SetWebViewClient(new WebViewClient());
            _webView.AddJavascriptInterface(new SignBridgeJsBridge(this, _localModelPath), "SignBridgeNative");
            _webView.LoadUrl("file:///android_asset/mobile_ui.html");
        }
    }

    private string EnsureLocalYoloModel()
    {
        try
        {
            var destDir = FilesDir?.AbsolutePath ?? CacheDir?.AbsolutePath ?? "";
            var destFile = Path.Combine(destDir, "yolo11n.pt");
            if (!File.Exists(destFile) && Assets != null)
            {
                using var input = Assets.Open("yolo11n.pt");
                using var output = File.Create(destFile);
                input.CopyTo(output);
            }
            return destFile;
        }
        catch
        {
            return "assets://yolo11n.pt";
        }
    }

    private sealed class SignBridgeChromeClient : WebChromeClient
    {
        public override void OnPermissionRequest(PermissionRequest? request)
        {
            request?.Grant(request.GetResources());
        }
    }

    private sealed class SignBridgeJsBridge : Java.Lang.Object
    {
        private readonly Context _context;
        private readonly string _modelPath;

        public SignBridgeJsBridge(Context context, string modelPath)
        {
            _context = context;
            _modelPath = modelPath;
        }

        [Export("GetYoloInfo")]
        [JavascriptInterface]
        public string GetYoloInfo()
        {
            try
            {
                long size = File.Exists(_modelPath) ? new FileInfo(_modelPath).Length : 5613764;
                using var stream = _context.Assets?.Open("data.yaml");
                if (stream != null)
                {
                    using var reader = new StreamReader(stream);
                    return $"SignBridge Local YOLOv11n Model\nPath: {_modelPath} ({size} bytes)\n\n" + reader.ReadToEnd();
                }
            }
            catch { }
            return $"SignBridge Local YOLOv11n Model ({_modelPath})";
        }
    }
}
