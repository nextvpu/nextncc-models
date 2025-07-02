# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None


a = Analysis(
    ['split_onnx_model_main.py'],
    pathex=['./UtilsFile'],
    binaries=[('UtilsFile', 'UtilsFile')],
    datas=[],
            
    hiddenimports=["numpy","numpy.random","onnxruntime","onnx_graphsurgeon","onnx"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    [],
    exclude_binaries=True,
    name='splitOnnxModel',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='splitOnnxModel',
)
