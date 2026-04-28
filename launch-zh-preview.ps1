$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$previewRoot = Join-Path (Split-Path -Parent $root) "zh-preview-official"
$widgetBase = Join-Path (Split-Path -Parent $root) "orange-widget-base"
$python = "C:\Users\Daniel\.conda\envs\orange3\python.exe"

if (-not (Test-Path $python)) {
    throw "Python not found: $python"
}

if (-not (Test-Path $previewRoot)) {
    throw "Preview directory not found: $previewRoot"
}

$env:PYTHONPATH = "$previewRoot;$widgetBase"

Push-Location $previewRoot
try {
    & $python -c "from AnyQt.QtCore import QSettings; lang = '\u7b80\u4f53\u4e2d\u6587'; settings = [QSettings(QSettings.IniFormat, QSettings.UserScope, 'biolab.si', 'Orange'), QSettings(QSettings.NativeFormat, QSettings.UserScope, 'biolab.si', 'Orange')]; [s.setValue('application/language', lang) or s.setValue('application/last-used-language', lang) or s.sync() for s in settings]"
    Start-Process -FilePath $python -ArgumentList "-m", "Orange.canvas" -WorkingDirectory $previewRoot
} finally {
    Pop-Location
}
