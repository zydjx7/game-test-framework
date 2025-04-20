@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM 文件操作工具 - 解决CMD中的路径和编码问题
REM 用法: file_tool.bat [操作] [路径]
REM 操作: list(列出), read(读取), find(查找)

if "%1"=="" goto :HELP
if "%2"=="" goto :HELP

set ACTION=%1
set PATH_ARG=%2

REM 把路径中的正斜杠替换为反斜杠
set SAFE_PATH=%PATH_ARG:/=\%

echo 正在执行: %ACTION% "%SAFE_PATH%"
echo.

if /i "%ACTION%"=="list" goto :LIST
if /i "%ACTION%"=="read" goto :READ
if /i "%ACTION%"=="find" goto :FIND
goto :HELP

:LIST
    echo 通过PowerShell安全列出文件:
    echo ----------------------------
    powershell -Command "if (Test-Path -Path '%SAFE_PATH%' -PathType Container) { Get-ChildItem -Force -Path '%SAFE_PATH%' | Format-Table Mode, LastWriteTime, Length, Name } else { Get-ChildItem -Force -Path '%SAFE_PATH%' }"
    exit /b 0

:READ
    echo 通过PowerShell安全读取文件:
    echo ----------------------------
    powershell -Command "if (Test-Path -Path '%SAFE_PATH%') { Get-Content -Path '%SAFE_PATH%' -Encoding UTF8 } else { Write-Host '错误: 文件不存在' -ForegroundColor Red }"
    exit /b 0

:FIND
    set /p SEARCH_TEXT=输入要查找的文本: 
    echo 通过PowerShell查找文本:
    echo ----------------------------
    powershell -Command "if (Test-Path -Path '%SAFE_PATH%') { Select-String -Path '%SAFE_PATH%' -Pattern '%SEARCH_TEXT%' } else { Write-Host '错误: 文件不存在' -ForegroundColor Red }"
    exit /b 0

:HELP
    echo.
    echo 文件操作工具 - 使用说明
    echo ===========================
    echo 用法: file_tool.bat [操作] [路径]
    echo.
    echo 可用操作:
    echo   list  - 列出文件或目录内容 (相当于dir /a)
    echo   read  - 读取文件内容 (相当于type)
    echo   find  - 在文件中查找文本 (相当于findstr)
    echo.
    echo 示例:
    echo   file_tool.bat list ".env"         - 列出当前目录中的.env文件
    echo   file_tool.bat list "Code"         - 列出Code目录内容
    echo   file_tool.bat read "Code\.env"    - 读取Code目录中的.env文件
    echo   file_tool.bat find "Code\.env"    - 在Code目录的.env文件中查找文本
    echo.
    exit /b 1 