# Building TraffiCount Pro for Windows

This document describes how to convert the existing Flask app into a
standalone Windows desktop application using **Nuitka** and **Inno Setup**.

## Overview

- The entry point for the executable is `app_wrapper.py` (renamed to `launcher`
in previous versions).
- The bundled executable (`TraffiCountPro.exe`) starts a local Flask server and
  opens the default browser on `http://127.0.0.1:5000`.
- All Python code, models, static files and dependencies are packaged; the
  user does **not** need to install Python, Flask, CUDA or any other library.
- CUDA-aware code is already present in `testApp1.py`; the launcher emits a
  warning if CUDA is not available.
- A Windows installer (`TraffiCountPro_Setup.exe`) is produced by Inno Setup.

----

## Step 1 – Prepare the environment

1. Install Python 3.10+ (used only for building).
2. Create a virtual environment and install the project's requirements:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   # make sure Nuitka and its dependencies are present
   pip install nuitka wheel
   # install pywin32 and other tools used by Nuitka on Windows
   pip install pywin32
   ```
3. Ensure the `modal` directory with the `.pt` model files and any other data
   are up to date.

## Step 2 – Create the Nuitka build script

A simple batch file (`build_nuitka.bat`) drives the build.  It compiles the
launcher, bundles all data files, and produces a single-file executable.

>Create or update `build_nuitka.bat` as follows:

```bat
@echo off
setlocal

REM path to the launcher script we just edited
set ENTRY=app_wrapper.py

REM output directory for the compilation
set OUTDIR=dist
set EXE=TraffiCountPro.exe

REM remove any previous build results
if exist %OUTDIR% rd /s /q %OUTDIR%

REM build with Nuitka in onefile standalone mode
python -m nuitka "%ENTRY%" \
    --standalone \
    --onefile \
    --output-dir=%OUTDIR% \
    --windows-disable-console \
    --follow-imports \
    --include-data-dir=templates=templates \
    --include-data-dir=static=static \
    --include-data-dir=modal=modal \
    --include-data-dir=lib=lib \
    --include-data-file=device.json=device.json \
    --include-data-file=user_config.json=user_config.json \
    --include-data-file=license.json=license.json \
    --plugin-enable=tk-inter

REM copy the final executable to its canonical name
move "%OUTDIR%\%ENTRY%.exe" "%OUTDIR%\%EXE%"

echo Build complete: %OUTDIR%\%EXE%
endlocal
```

>Adjust paths as necessary.  The `--include-data-*` arguments ensure that the
Flask templates, CSS/JS, model weights and other non-Python assets are
packaged.

>**Note:** Nuitka will compile the Python source to C and then to machine
code; the original `.py` files are not shipped, which protects your source.

## Step 3 – Verify the standalone executable

1. Run `build_nuitka.bat` from the project root.
2. After the build completes, launch `dist\TraffiCountPro.exe` on the build
   machine.  A console may briefly appear (it is hidden by the `--windows-disable-console` flag).
3. The default browser should open `http://127.0.0.1:5000` and the app should
   behave normally.
4. Test features such as video upload, license activation, and GPU detection.

If you need to debug issues during compilation, omit `--onefile` to get a
`dist\<ENTRY>.dist` directory containing log files.

## Step 4 – Create the Inno Setup installer

1. [Download and install Inno Setup](https://jrsoftware.org/isinfo.php).
2. Create an ISS script named `TraffiCountPro.iss` with the following contents
   (place it in the project root):

```ini
[Setup]
AppName=TraffiCount Pro
AppVersion=1.0.0
DefaultDirName={pf}\TraffiCount Pro
DefaultGroupName=TraffiCount Pro
OutputDir=installer
OutputBaseFilename=TraffiCountPro_Setup
Compression=lzma2
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: desktopicon; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "dist\TraffiCountPro.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\TraffiCount Pro"; Filename: "{app}\TraffiCountPro.exe"
Name: "{commondesktop}\TraffiCount Pro"; Filename: "{app}\TraffiCountPro.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\TraffiCountPro.exe"; Description: "Launch TraffiCount Pro"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{app}\TraffiCountPro.exe"
```

3. Run the Inno Setup compiler on the script; the resulting
   `installer\TraffiCountPro_Setup.exe` is the installer you distribute.

## Important points for licensing and models

* Because Nuitka bundles the Python interpreter and your source code, end
  users cannot simply open `.py` files or see raw model weights – they are
  embedded inside the executable.  For additional obfuscation you can apply
  the `--python-forbid-recompile` flag to prevent reverse‑engineering.
* The `.pt` files under `modal/` will be copied verbatim into the EXE; they
  are not readable until extracted by the running program.  If you require
  stronger protection, consider encrypting them and decrypting at runtime.

## CUDA handling

The compiled program does not ship CUDA itself – the end user must have a
compatible NVIDIA driver and optionally the CUDA runtime installed.  During
startup the launcher prints a warning if `torch.cuda.is_available()` is
false.  You can extend the check to show a message box or disable GPU‑related
functionality.

## Distribution workflow

1. Build the executable with `build_nuitka.bat` on a clean Windows machine
   that has the same architecture as your target audience (e.g. x64).
2. Run Inno Setup to generate the installer.
3. Test the installer on an unprepared VM to ensure no dependencies are
   required by the user.

---

This procedure gives you a single-file, installable, protected Windows
application that behaves identically to the Flask development server but
requires zero setup from the customer.