from split_onnx_model import split_onnx
from add_customOp_onnx_model import addSubGraph
from UtilsFile.Utils import ToVariableName
import argparse
import numpy as np
import os, sys


def parse_list_str(value):
    return [str(x) for x in value.split(",")]


def parse_list_np(value):
    return [x for x in value.split(",")]


def parse_list_int(values):
    finallyLst = []
    for value in values.split(";"):
        finallyLst.append([int(x) for x in value.split(",")])
    return finallyLst

def SplitOnnxMain(args):
    args.savePath = os.path.join(args.modelPath, args.modelName+'_bakbone.onnx')

    output_name = []
    new_bottom_name_name = [
        ToVariableName(Name, withPrefix=False) for Name in args.bottomName
    ]
    new_top_mode_name = [
        ToVariableName(Name, withPrefix=False) for Name in args.topName
    ]
    model_path = r"{}\{}.onnx".format(args.modelPath, args.modelName)
    if args.savePath != "":
        save_path = args.savePath
    else:
        save_path = r"{}\{}_{}_{}.onnx".format(
            args.modelPath,
            args.modelName,
            "_".join(new_bottom_name_name),
            "_".join(new_top_mode_name),
        )
    # save_path = r"{}\{}_SubGraph.onnx".format(modelPath, model_name)
    try:
        print("====> args: ", args)
        print("====> Start Split Model {}~".format(args.modelName))
        split_onnx(
            model_path,
            save_path,
            args.bottomName,
            args.topName,
            args.modelInputName,
            output_name,
            args.inputShapes,
            args.inputTypes,
            args.outputShapes,
            args.outputTypes,
        )
        print("====> End Split Model {}~".format(args.modelName))
        sys.exit(0)
    except Exception as e:
        print("====> Split Model {} Failed~".format(args.modelName))
        import traceback  
        stack_trace = traceback.format_exc() 
        print("====> Exception: ", stack_trace)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="====>split or add onnx model~")

    # Positional arguments.
    parser.add_argument('--func_selection', type=str, 
                        help=
                        'Function selection: \n \
                            1: Add post-processing custom operators to the original onnx model; \
                        '
                        , default='0')
    parser.add_argument("--modelName", type=str, required=False, help="model_sim")
    parser.add_argument(
        "--modelPath",
        type=str,
        required=False,
        help=r"D:\nextvpu\product\compiler-v3_ori\output\test\NDS_Regression_20231017\modify\n16x\onnx\DBNet_r50dcn\pc\model",
    )
    parser.add_argument(
        "--oriModel",
        type=str,
        required=False,
        help=r"D:\NextVPU_Self\PPQ\models\front_rear_sense2d\20250606\yolov8n-2dod-nc4-20250605173848_backbone_mAP0_545_int8_backbone_mAP0_526.onnx",
    )
    parser.add_argument(
        "--postModel",
        type=str,
        required=False,
        help=r"D:\NextVPU_Self\WorkList\WH\four_model_test\Yolov8\Yolov8.onnx",
    )
    parser.add_argument(
        "--modelInputName",
        type=parse_list_str,
        required=False,
        default=[],
        help="output_0,output_1,output_2,output_3",
    )
    parser.add_argument(
        "--bottomName",
        type=parse_list_str,
        required=False,
        default=[],
        help='/NonZero_2,/Gather_6,/Gather_7",/Gather_4,/Gather_5',
    )
    parser.add_argument(
        "--topName",
        type=parse_list_str,
        required=False,
        default=[],
        help='/NonZero_2,/Gather_6,/Gather_7",/Gather_4,/Gather_5',
    )
    parser.add_argument(
        "--inputShapes",
        type=parse_list_int,
        required=False,
        default=[],
        help="1,3,224,224;1,3,640,640",
    )
    parser.add_argument(
        "--inputTypes",
        type=parse_list_np,
        required=False,
        default=[],
        help="float32,float32]",
    )
    parser.add_argument(
        "--outputShapes",
        type=parse_list_int,
        required=False,
        default=[],
        help="1,3,224,224;1,3,640,640",
    )
    parser.add_argument(
        "--outputTypes",
        type=parse_list_np,
        required=False,
        default=[],
        help="float32,float32",
    )
    parser.add_argument(
        "--savePath", type=str, required=False, default="", help="save output path"
    )

    args = parser.parse_args()
    ############################################################# args.func_selection == '0' #############################################################
    args.func_selection = '0'
    args.modelName = "yolov8semi3d_backbone_int8_label4_20250624"
    args.modelPath = r"D:\NextVPU_Self\WorkList\HYG\limin_yolov8semi3d\four_classes"
    args.modelOutPath = r"D:\NextVPU_Self\WorkList\HYG\limin_yolov8semi3d\four_classes"
    args.modelInputName = ["images"]  # , "imgBgr"
    args.bottomName = [
        # "PPQ_Operation_0",
        "PPQ_Operation_0"
    ]
    args.topName = [
        "/model.22/cv4.0/cv4.0.2/Conv",
        "/model.22/cv4.1/cv4.1.2/Conv",
        "/model.22/cv4.2/cv4.2.2/Conv",
        "/model.22/cv2.0/cv2.0.2/Conv",
        "/model.22/cv3.0/cv3.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv",
        "/model.22/cv3.1/cv3.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv",
        "/model.22/cv3.2/cv3.2.2/Conv",
        # "/cls/cls.3/Gemm"
        # "/cls/cls.3/Gemm_row",
        # "/cls/cls.3/Gemm_col"
    ]
    # args.inputShapes = [[1, 1, 1280, 1280]]  # , [1, 3, 1513, 1271], [1, 3, 1280, 1280]
    # args.inputTypes = [np.int8, np.int8, np.int8]
    args.outputTypes = [
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ]
    ############################################################# args.func_selection == '1' #############################################################
    args.func_selection = '1'
    args.oriModel = r"D:\NextVPU_Self\PPQ\models\front_rear_sense2d\20250606\yolov8n-2dod-nc4-20250605173848_backbone_mAP0_545_int8_backbone_mAP0_526.onnx"
    args.postModel = r"D:\NextVPU_Self\WorkList\WH\four_model_test\Yolov8\Yolov8.onnx"
    args.savePath = r"D:\NextVPU_Self\WorkList\WH\four_model_test\yolov8Backbone_yolov8Decode\yolov8Backbone_yolov8Decode.onnx"
    args.bottomName = [
        "/model.22/cv2.0/cv2.0.2/Conv_output_0",
        "/model.22/cv2.1/cv2.1.2/Conv_output_0",
        "/model.22/cv2.2/cv2.2.2/Conv_output_0",
        "/model.22/cv3.0/cv3.0.2/Conv_output_0",
        "/model.22/cv3.1/cv3.1.2/Conv_output_0",
        "/model.22/cv3.2/cv3.2.2/Conv_output_0"
    ]
    args.topName = [
        "Yolov8_input0",
        "Yolov8_input3",
        "Yolov8_input1",
        "Yolov8_input4",
        "Yolov8_input2",
        "Yolov8_input5"
    ]
    ##########################################################################################################################
    if args.func_selection == '0':
        res = SplitOnnxMain(args)
    elif args.func_selection == '1':
        res = addSubGraph(args.oriModel, args.postModel, args.savePath, args.bottomName, args.topName)
    
