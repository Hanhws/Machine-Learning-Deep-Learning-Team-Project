@echo off
title 심사 기간 전원 설정 (관리자 권한 필요)

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo   [!] 관리자 권한이 아닙니다.
    echo.
    echo   이 파일을 마우스 우클릭 - "관리자 권한으로 실행" 으로 열어주세요.
    echo.
    pause
    exit /b 1
)

echo.
echo   ================================================
echo    심사 기간 동안 노트북이 잠들지 않게 설정합니다
echo   ================================================
echo.

echo   [1/4] 절전 모드 끄기 (전원 연결 시)
powercfg /change standby-timeout-ac 0

echo   [2/4] 화면 끄기 안 함
powercfg /change monitor-timeout-ac 0

echo   [3/4] 최대 절전 모드 끄기
powercfg /change hibernate-timeout-ac 0

echo   [4/4] 덮개를 닫아도 아무 동작 안 함
powercfg /setacvalueindex SCHEME_CURRENT 4f971e89-eebd-4455-a8de-9e59040e7347 5ca83367-6e45-459f-a27b-476b1d01c936 0
powercfg /setactive SCHEME_CURRENT

echo.
echo   [+] 로그온 시 WSL 자동 시작 등록
schtasks /Create /TN "traffic-wsl" /TR "wsl.exe -d Ubuntu-24.04 -u root /bin/true" /SC ONLOGON /RL HIGHEST /F > nul
if %errorLevel% equ 0 (echo       traffic-wsl 등록됨) else (echo       [!] 등록 실패)

echo.
echo   ================================================
echo    적용된 설정 확인
echo   ================================================
echo.
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | findstr /C:"현재 AC" /C:"Current AC"
echo.
echo   전부 0x00000000 이면 정상입니다 (0 = 안 함).
echo.
echo   남은 수동 작업:
echo     설정 - Windows Update - 업데이트 일시 중지 - 5주
echo.
echo   주의: 배터리로 쓰면 그대로 잠듭니다. 어댑터를 꽂아 두세요.
echo.
pause
