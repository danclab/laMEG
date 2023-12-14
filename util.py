import json


def get_spm_path():
    with open('settings.json') as settings_file:
        parameters = json.load(settings_file)
    spm_path = parameters["spm_path"]
    return spm_path