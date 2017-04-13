
if (Test-Path dist) {
    Remove-Item -Recurse -Force dist
}

python setup.py clean --all

cmd.exe /c @"
.ci_tools\appveyor\build.cmd python setup.py $env:BUILD_GLOBAL_OPTIONS bdist_wheel --dist-dir dist
"@

if ($LastExitCode -ne 0) { throw "Last command exited with non-zero code." }
