@ECHO OFF

pushd %~dp0

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "html" goto html

:help
echo.Please use `make ^<target^>` where ^<target^> is one of
echo.  html       to make standalone HTML files
goto end

:html
sphinx-build -b html . _build/html
echo.
echo.Build finished. The HTML pages are in _build/html.
goto end

:end
popd
