from datetime import datetime

class _Vars:
    current_datetime = datetime.now().strftime("%d-%m_%H-%M")
    date = current_datetime.split("_")[0]
    strain = 'SB5'
    isConfig, initState, ports = False, False, []
    activeDevices, initRun, initTime = {}, False, 0
    dataSaveTimer, saveTime = 5000, 0
    notConnected = False
    config = {}