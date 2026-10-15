Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "f:\Random\WORK\quantlab"
WshShell.Run "f:\Random\WORK\quantlab\venv\Scripts\pythonw.exe -m solana_sniper.bot_solana --loop", 0, False
