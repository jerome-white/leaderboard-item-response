from datetime import datetime

def hf_datetime(dstring):
    if dstring.find(':') >= 0:
        dstring = dstring.replace(':', '-')

    return datetime.strptime(dstring, '%Y-%m-%dT%H-%M-%S.%f')
