from UtilsFile.Utils import reset_ir_version
import onnx
import onnx_graphsurgeon as gs


def addSubGraph(oriModel, postModel, save_path, bottomNames, topNames):
    # ori graph
    ori_onnx_file = onnx.load(oriModel)
    ori_graph = gs.import_onnx(ori_onnx_file)
    # dataconvertType graph
    post_onnx_file_0 = onnx.load(postModel)
    post_graph_0 = gs.import_onnx(post_onnx_file_0)

    for postnode in post_graph_0.nodes:
        ori_graph.nodes.append(postnode)


    connTopTnsrName = topNames
    connBottomTnsrName = bottomNames

    for idx, name in enumerate(connBottomTnsrName):
        for postnode in ori_graph.nodes:
            if postnode.name == post_graph_0.nodes[0].name:
                for orinode in ori_graph.nodes:
                    if name in [output.name for output in orinode.outputs]:
                        print("====> connect {} -> {}".format(name, postnode.inputs[idx]))
                        for idx_2, output_2 in enumerate(orinode.outputs):
                            if output_2.name == name:
                                del orinode.outputs[idx_2]
                                orinode.outputs.insert(idx_2, postnode.inputs[idx])
    
    delGraphOutputNum = len(ori_graph.outputs)
    for d in range(delGraphOutputNum):
        ori_graph.outputs.pop()
    
    for post_graph_node in post_graph_0.nodes:
        for post_graph_output in post_graph_node.outputs:
            print("====> add {} to graph output".format(post_graph_output.name))
            ori_graph.outputs.append(post_graph_node.outputs[0])

    ori_graph.cleanup().toposort()
    onnx.save(gs.export_onnx(ori_graph), save_path)
    reset_ir_version(save_path)

# if __name__ == '__main__':
#     oriModel = r"D:\NextVPU_Self\PPQ\models\front_rear_sense2d\20250606\yolov8n-2dod-nc4-20250605173848_backbone_mAP0_545_int8_backbone_mAP0_526.onnx"
#     addModel = r"D:\NextVPU_Self\WorkList\WH\four_model_test\Yolov8\Yolov8.onnx"
#     save_path = r"D:\NextVPU_Self\WorkList\WH\four_model_test\yolov8Backbone_yolov8Decode\yolov8Backbone_yolov8Decode.onnx"
#     bottomNames = [
#         "/model.22/cv2.0/cv2.0.2/Conv_output_0",
#         "/model.22/cv3.0/cv3.0.2/Conv_output_0",
#         "/model.22/cv2.1/cv2.1.2/Conv_output_0",
#         "/model.22/cv3.1/cv3.1.2/Conv_output_0",
#         "/model.22/cv2.2/cv2.2.2/Conv_output_0",
#         "/model.22/cv3.2/cv3.2.2/Conv_output_0"
#     ]
#     topNames = [
#         "Yolov8_input0",
#         "Yolov8_input3",
#         "Yolov8_input1",
#         "Yolov8_input4",
#         "Yolov8_input2",
#         "Yolov8_input5"
#     ]
#     addSubGraph(oriModel, addModel, save_path, bottomNames, topNames)
