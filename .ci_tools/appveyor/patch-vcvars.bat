@echo off
setlocal
:: The VS 2010 64-bit toolchain is missing a vcvars64.bat that distutils is
:: using to query the compiler.
set "vcvars=C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\bin\amd64\vcvars64.bat"
if not exist "%vcvars%" (
    echo CALL "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64 /Release
) > "%vcvars%"
endlocal
@echo on
