import re
import onnx


def ToVariableName(nameStr, withPrefix=True, Overlength=True) :
    if len(nameStr)>50 and Overlength==False:
        splitSymLst = [':', '\.', '/', '-', ';', '_']
        shortName = 'TS_'
        _splitted = False
        for i in range(len(nameStr)):
            if nameStr[i] in splitSymLst:
                if _splitted is False:
                    shortName += '_'
                shortName += nameStr[i+1]
                _splitted = True
            if _splitted is False:
                shortName += nameStr[i]
            elif nameStr[i].isdigit():
                shortName += nameStr[i]
        return shortName
    elif nameStr == '':
        return nameStr
    else:
        s = re.sub(':', '_{:02x}_'.format(ord(':')), nameStr)
        s = re.sub('\.', '_{:02x}_'.format(ord('.')), s)
        s = re.sub('/', '_{:02x}_'.format(ord('/')), s)
        s = re.sub('-', '_{:02x}_'.format(ord('-')), s)
        s = re.sub(';', '_{:02x}_'.format(ord(';')), s)
        if withPrefix:
            s = "Tensor_"+s if s[0].isdigit() else s
            s = "Tensor"+s if s[0]=='_' else s
        return s

def reset_ir_version(path):
    _opsetVersion2IRVersion = {
        1:3,
        5:3,
        6:3,
        7:3,
        8:3,
        9:4,
        10:5,
        11:6,
        12:7,
        13:7,
        14:7,
        15:8,
        20:9,
    }

    model = onnx.load(path)
    # model.opset_import[0].version =14

    opset = model.opset_import[0].version

    model.ir_version = _opsetVersion2IRVersion[opset]

    # onnx.checker.check_model(model)
    onnx.save(model, path)
    print("====> reset ir_version to {}".format(model.ir_version))