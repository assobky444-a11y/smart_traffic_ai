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