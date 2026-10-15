Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "f:\Random\WORK\quantlab"
WshShell.Run "f:\Random\WORK\quantlab\venv\Scripts\pythonw.exe -m sol_scalper.bot_scalper --loop", 0, False
