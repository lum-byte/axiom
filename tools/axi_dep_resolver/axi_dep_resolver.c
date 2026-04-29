#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <string.h>

#define AXI_DEP_RESOLVER_VERSION L"1.0.5"

static int is_elevated(void);
static int relaunch_elevated(void);
static int write_installer_script(wchar_t *script_path, DWORD script_path_cap);
static int run_powershell_script(const wchar_t *script_path);
static void show_last_error(const wchar_t *prefix);

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command_line, int show_command) {
    (void)instance;
    (void)previous;
    (void)show_command;

    if (wcsstr(command_line, L"--elevated") == NULL && !is_elevated()) {
        int answer = MessageBoxW(
            NULL,
            L"AXIOM dependency resolver can configure PATH and download official runtime installers.\n\n"
            L"Windows may ask for administrator permission. Continue?",
            L"AXIOM dependency resolver",
            MB_ICONINFORMATION | MB_YESNO | MB_DEFBUTTON1
        );
        if (answer != IDYES) {
            return 0;
        }
        if (!relaunch_elevated()) {
            show_last_error(L"Could not request administrator permission");
            return 2;
        }
        return 0;
    }

    int accepted = MessageBoxW(
        NULL,
        L"AXIOM will download official redistributable/bootstrapper files from vendor URLs and write a manifest.\n\n"
        L"It will not silently accept third-party license terms or install GPU/toolchain packages for you.\n"
        L"Review each vendor installer before running it.\n\n"
        L"Proceed?",
        L"AXIOM dependency resolver terms",
        MB_ICONWARNING | MB_YESNO | MB_DEFBUTTON2
    );
    if (accepted != IDYES) {
        return 0;
    }

    wchar_t script_path[MAX_PATH * 2];
    if (!write_installer_script(script_path, (DWORD)(sizeof(script_path) / sizeof(script_path[0])))) {
        show_last_error(L"Could not write dependency resolver script");
        return 3;
    }
    if (!run_powershell_script(script_path)) {
        show_last_error(L"Dependency resolver script failed");
        return 4;
    }

    MessageBoxW(
        NULL,
        L"AXIOM dependency resolver finished.\n\n"
        L"Restart your terminal or AXIOM app so PATH changes are visible.",
        L"AXIOM dependency resolver",
        MB_ICONINFORMATION | MB_OK
    );
    return 0;
}

static int is_elevated(void) {
    BOOL is_member = FALSE;
    PSID admin_group = NULL;
    SID_IDENTIFIER_AUTHORITY nt_authority = SECURITY_NT_AUTHORITY;
    if (!AllocateAndInitializeSid(
            &nt_authority,
            2,
            SECURITY_BUILTIN_DOMAIN_RID,
            DOMAIN_ALIAS_RID_ADMINS,
            0,
            0,
            0,
            0,
            0,
            0,
            &admin_group)) {
        return 0;
    }
    if (!CheckTokenMembership(NULL, admin_group, &is_member)) {
        is_member = FALSE;
    }
    FreeSid(admin_group);
    return is_member ? 1 : 0;
}

static int relaunch_elevated(void) {
    wchar_t exe_path[MAX_PATH * 2];
    if (GetModuleFileNameW(NULL, exe_path, (DWORD)(sizeof(exe_path) / sizeof(exe_path[0]))) == 0) {
        return 0;
    }
    HINSTANCE result = ShellExecuteW(NULL, L"runas", exe_path, L"--elevated", NULL, SW_SHOWNORMAL);
    return ((INT_PTR)result) > 32;
}

static int write_installer_script(wchar_t *script_path, DWORD script_path_cap) {
    wchar_t temp_dir[MAX_PATH * 2];
    DWORD temp_len = GetTempPathW((DWORD)(sizeof(temp_dir) / sizeof(temp_dir[0])), temp_dir);
    if (temp_len == 0 || temp_len >= (DWORD)(sizeof(temp_dir) / sizeof(temp_dir[0]))) {
        return 0;
    }
    if (swprintf(script_path, script_path_cap, L"%saxi_dep_resolver_%lu.ps1", temp_dir, GetTickCount()) < 0) {
        return 0;
    }

    FILE *file = _wfopen(script_path, L"wb");
    if (file == NULL) {
        return 0;
    }

    const char *script =
        "$ErrorActionPreference = 'Continue'\r\n"
        "$ProgressPreference = 'Continue'\r\n"
        "$root = Join-Path $env:LOCALAPPDATA 'Axiom\\deps'\r\n"
        "$downloads = Join-Path $root 'downloads'\r\n"
        "New-Item -ItemType Directory -Force -Path $downloads | Out-Null\r\n"
        "$links = @(\r\n"
        "  @{Name='Microsoft Visual C++ Redistributable x64'; Url='https://aka.ms/vs/17/release/vc_redist.x64.exe'; File='vc_redist.x64.exe'; Direct=$true},\r\n"
        "  @{Name='Visual Studio Build Tools bootstrapper'; Url='https://aka.ms/vs/17/release/vs_BuildTools.exe'; File='vs_BuildTools.exe'; Direct=$true},\r\n"
        "  @{Name='Rustup x64'; Url='https://win.rustup.rs/x86_64'; File='rustup-init.exe'; Direct=$true},\r\n"
        "  @{Name='Python downloads'; Url='https://www.python.org/downloads/windows/'; File='python-downloads.url'; Direct=$false},\r\n"
        "  @{Name='Git for Windows'; Url='https://git-scm.com/download/win'; File='git-for-windows.url'; Direct=$false},\r\n"
        "  @{Name='Node.js downloads'; Url='https://nodejs.org/en/download'; File='nodejs-download.url'; Direct=$false},\r\n"
        "  @{Name='Go downloads'; Url='https://go.dev/dl/'; File='go-download.url'; Direct=$false},\r\n"
        "  @{Name='CUDA downloads'; Url='https://developer.nvidia.com/cuda-downloads'; File='cuda-download.url'; Direct=$false},\r\n"
        "  @{Name='Tor Expert Bundle'; Url='https://www.torproject.org/download/tor/'; File='tor-download.url'; Direct=$false}\r\n"
        ")\r\n"
        "$manifest = @()\r\n"
        "$i = 0\r\n"
        "foreach ($link in $links) {\r\n"
        "  $i++\r\n"
        "  Write-Progress -Activity 'Resolving AXIOM dependencies' -Status $link.Name -PercentComplete (($i / $links.Count) * 100)\r\n"
        "  $target = Join-Path $downloads $link.File\r\n"
        "  $entry = [ordered]@{ name=$link.Name; url=$link.Url; target=$target; direct=$link.Direct; ok=$false; error=$null }\r\n"
        "  try {\r\n"
        "    if ($link.Direct) {\r\n"
        "      Invoke-WebRequest -Uri $link.Url -Method Head -MaximumRedirection 8 -TimeoutSec 30 | Out-Null\r\n"
        "      Invoke-WebRequest -Uri $link.Url -OutFile $target -MaximumRedirection 8 -TimeoutSec 300\r\n"
        "    } else {\r\n"
        "      Set-Content -Path $target -Encoding ASCII -Value ('[InternetShortcut]`r`nURL=' + $link.Url + '`r`n')\r\n"
        "      Start-Process $link.Url\r\n"
        "    }\r\n"
        "    $entry.ok = Test-Path $target\r\n"
        "  } catch {\r\n"
        "    $entry.error = $_.Exception.Message\r\n"
        "  }\r\n"
        "  $manifest += [pscustomobject]$entry\r\n"
        "}\r\n"
        "$candidatePaths = @(\r\n"
        "  \"$env:USERPROFILE\\.cargo\\bin\",\r\n"
        "  \"$env:ProgramFiles\\Git\\cmd\",\r\n"
        "  \"$env:ProgramFiles\\nodejs\",\r\n"
        "  \"$env:ProgramFiles\\Go\\bin\",\r\n"
        "  \"$env:ProgramFiles\\CMake\\bin\",\r\n"
        "  \"$env:ProgramFiles\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\",\r\n"
        "  \"$env:ProgramFiles\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64\",\r\n"
        "  \"$env:ProgramData\\chocolatey\\bin\"\r\n"
        ")\r\n"
        "$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')\r\n"
        "if ($null -eq $userPath) { $userPath = '' }\r\n"
        "$parts = $userPath.Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)\r\n"
        "$changed = $false\r\n"
        "foreach ($path in $candidatePaths) {\r\n"
        "  if ((Test-Path $path) -and ($parts -notcontains $path)) {\r\n"
        "    $parts += $path\r\n"
        "    $changed = $true\r\n"
        "  }\r\n"
        "}\r\n"
        "if ($changed) { [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User') }\r\n"
        "$manifestPath = Join-Path $root 'dependency-manifest.json'\r\n"
        "$manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $manifestPath\r\n"
        "Write-Progress -Activity 'Resolving AXIOM dependencies' -Completed\r\n"
        "Write-Host \"AXIOM dependency resolver 1.0.5 complete\"\r\n"
        "Write-Host \"Downloads: $downloads\"\r\n"
        "Write-Host \"Manifest:  $manifestPath\"\r\n"
        "Write-Host \"Restart terminal/app before testing PATH changes.\"\r\n"
        "Read-Host 'Press Enter to close'\r\n";

    size_t written = fwrite(script, 1, strlen(script), file);
    fclose(file);
    return written == strlen(script);
}

static int run_powershell_script(const wchar_t *script_path) {
    wchar_t params[MAX_PATH * 3];
    if (swprintf(
            params,
            (DWORD)(sizeof(params) / sizeof(params[0])),
            L"-NoProfile -ExecutionPolicy Bypass -File \"%s\"",
            script_path) < 0) {
        return 0;
    }

    SHELLEXECUTEINFOW info;
    ZeroMemory(&info, sizeof(info));
    info.cbSize = sizeof(info);
    info.fMask = SEE_MASK_NOCLOSEPROCESS;
    info.lpVerb = L"open";
    info.lpFile = L"powershell.exe";
    info.lpParameters = params;
    info.nShow = SW_SHOWNORMAL;
    if (!ShellExecuteExW(&info)) {
        return 0;
    }
    WaitForSingleObject(info.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(info.hProcess, &code);
    CloseHandle(info.hProcess);
    return code == 0;
}

static void show_last_error(const wchar_t *prefix) {
    wchar_t message[1024];
    DWORD err = GetLastError();
    swprintf(message, (DWORD)(sizeof(message) / sizeof(message[0])), L"%s.\n\nWindows error: %lu", prefix, err);
    MessageBoxW(NULL, message, L"AXIOM dependency resolver", MB_ICONERROR | MB_OK);
}
