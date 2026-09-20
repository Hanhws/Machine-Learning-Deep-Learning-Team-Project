# 윈도우 쪽 준비를 한 번에 한다.
#   - WSL2 + Ubuntu 24.04 설치
#   - WSL 메모리 할당(.wslconfig)
#   - 절전 / 화면 끄기 / 덮개 닫기 끄기   ← 노트북이 자면 사이트가 죽는다
#   - 로그온 시 WSL 자동 시작 등록
#
# 관리자 PowerShell 에서:
#   irm https://raw.githubusercontent.com/Hanhws/traffic-deploy/main/windows-setup.ps1 | iex

$ErrorActionPreference = 'Stop'

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "   ! $m" -ForegroundColor Yellow }
function Ok($m)  { Write-Host "   $m" -ForegroundColor Green }

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
      ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Host "관리자 권한 PowerShell 에서 실행하세요." -ForegroundColor Red
  Write-Host "  시작 메뉴 → PowerShell 우클릭 → 관리자 권한으로 실행" -ForegroundColor Red
  return
}

$distro = 'Ubuntu-24.04'

# ─────────────────────────────────────────────────────────────
Say "1) WSL2 + $distro"
$needReboot = $false
$installed = (wsl.exe --list --quiet 2>$null) -replace "`0", ""
if ($installed -match [regex]::Escape($distro)) {
  Ok "$distro 이미 설치됨"
} else {
  Write-Host "   설치 중... (몇 분 걸립니다)"
  wsl.exe --install -d $distro --no-launch
  if ($LASTEXITCODE -ne 0) {
    # 구버전 윈도우는 --no-launch 를 모른다
    wsl.exe --install -d $distro
  }
  $needReboot = $true
  Ok "설치 요청 완료"
}
wsl.exe --set-default-version 2 | Out-Null

# ─────────────────────────────────────────────────────────────
Say "2) WSL 메모리 할당 (.wslconfig)"
# 전체 RAM 의 3/4 을 WSL 에 준다. 윈도우 몫으로 최소 4GB 는 남긴다.
$totalGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$wslGB = [math]::Max(8, $totalGB - 4)
$wslconfig = Join-Path $env:USERPROFILE '.wslconfig'
@"
[wsl2]
memory=${wslGB}GB
swap=8GB
"@ | Set-Content -Path $wslconfig -Encoding ASCII
Ok "RAM ${totalGB}GB 중 ${wslGB}GB 를 WSL 에 할당 → $wslconfig"

# ─────────────────────────────────────────────────────────────
Say "3) 절전 끄기 (전원 연결 시)"
powercfg /change standby-timeout-ac 0      # 절전 모드 안 함
powercfg /change monitor-timeout-ac 0      # 화면 끄기 안 함
powercfg /change hibernate-timeout-ac 0    # 최대 절전 안 함
# 덮개를 닫아도 아무 것도 안 함
$subButtons = '4f971e89-eebd-4455-a8de-9e59040e7347'
$lidAction  = '5ca83367-6e45-459f-a27b-476b1d01c936'
powercfg /setacvalueindex SCHEME_CURRENT $subButtons $lidAction 0
powercfg /setactive SCHEME_CURRENT
Ok "절전·화면끄기·덮개닫기 모두 해제 (전원 연결 시)"
Warn "배터리로 쓰면 그대로 잠듭니다. 심사 기간에는 어댑터를 꽂아 두세요."

# ─────────────────────────────────────────────────────────────
Say "4) 로그온하면 WSL 자동 시작"
# WSL 은 누군가 열어야 뜬다. 재부팅돼도 알아서 살아나게 등록해 둔다.
schtasks /Create /TN "traffic-wsl" /TR "wsl.exe -d $distro -u root /bin/true" `
         /SC ONLOGON /RL HIGHEST /F | Out-Null
Ok "작업 스케줄러에 'traffic-wsl' 등록"

# ─────────────────────────────────────────────────────────────
Say "5) GPU 드라이버"
$gpu = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' })
if ($gpu) {
  Ok "$($gpu.Name) · 드라이버 $($gpu.DriverVersion)"
  Warn "WSL 안에는 드라이버를 설치하지 마세요. 윈도우 드라이버가 그대로 넘어갑니다."
} else {
  Warn "NVIDIA GPU 를 못 찾았습니다. CPU 로 돌게 됩니다(느림)."
}

# ─────────────────────────────────────────────────────────────
Write-Host "`n───────────────────────────────────────────" -ForegroundColor DarkGray
if ($needReboot) {
  Write-Host @"

  재부팅이 필요합니다.

  재부팅 뒤에 할 일:
   1. 시작 메뉴에서 'Ubuntu 24.04' 실행 → 사용자 이름·비밀번호 정하기
   2. 그 우분투 창에 아래 한 줄을 붙여넣기

      git clone https://github.com/Hanhws/traffic-deploy.git ~/deploy && bash ~/deploy/install.sh

"@ -ForegroundColor White
  $a = Read-Host "지금 재부팅할까요? (y/N)"
  if ($a -eq 'y') { Restart-Computer -Force }
} else {
  Write-Host @"

  윈도우 쪽 준비 끝.

  다음: 시작 메뉴 → 'Ubuntu 24.04' 실행 후 아래 한 줄

      git clone https://github.com/Hanhws/traffic-deploy.git ~/deploy && bash ~/deploy/install.sh

"@ -ForegroundColor White
}

Write-Host "  남은 수동 작업 1개:" -ForegroundColor Yellow
Write-Host "  설정 → Windows Update → '업데이트 일시 중지' 를 최대(5주)로 걸어 두세요." -ForegroundColor Yellow
Write-Host "  자동 재부팅되면 WSL 이 내려가고 사이트가 죽습니다.`n" -ForegroundColor Yellow
