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

    protected override void OnCreate(Bundle? savedInstanceState)
    {
        base.OnCreate(savedInstanceState);
        SetContentView(Resource.Layout.activity_main);

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
            _webView.AddJavascriptInterface(new SignBridgeJsBridge(this), "SignBridgeNative");
            _webView.LoadUrl("file:///android_asset/mobile_ui.html");
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
        public SignBridgeJsBridge(Context context) => _context = context;

        [Export("GetYoloInfo")]
        [JavascriptInterface]
        public string GetYoloInfo()
        {
            try
            {
                using var stream = _context.Assets?.Open("data.yaml");
                if (stream != null)
                {
                    using var reader = new StreamReader(stream);
                    return "SignBridge YOLOv11n Model (yolo11n.pt)\n\n" + reader.ReadToEnd();
                }
            }
            catch { }
            return "SignBridge YOLOv11n Model (20 Classes)";
        }
    }
}
