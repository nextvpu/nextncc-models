from hashlib import new
import onnx_graphsurgeon as gs
import numpy as np
import onnx
import copy


def reset_ir_version(path):
    _opsetVersion2IRVersion = {
        1: 3,
        5: 3,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
        10: 5,
        11: 6,
        12: 7,
        13: 7,
        14: 7,
        15: 8,
        17: 9,
        18: 9,
    }

    model = onnx.load(path)

    opset = model.opset_import[0].version
    model.ir_version = _opsetVersion2IRVersion[opset]

    # onnx.checker.check_model(model)
    onnx.save(model, path)


def split_onnx(
    model_path,
    save_path,
    bottom_node_name,
    top_mode_name,
    model_input_name,
    model_output_name=[],
    inputShapes=[],
    inputTypes=[],
    outputShapes=[],
    outputTypes=[],
):
    graph = gs.import_onnx(onnx.load(model_path))

    # bottom_name = [node for node in graph.nodes if node.name == bottom_node_name][0]
    bottom_nodes = []
    for node in graph.nodes:
        for node_b in bottom_node_name:
            if node.name == node_b:
                bottom_nodes.append(node)
        if len(bottom_nodes) == len(bottom_node_name):
            break

    #############
    # for i in range(len(bottom_nodes)):
    #     bottom_nodes[i].inputs[0].dtype = np.float32
    #############

    #############
    # for i in range(len(bottom_nodes)):
    #     bottom_nodes[i].inputs[0].dtype = np.float32
    #############

    ori_bottom_nodes_0_input_num = len(bottom_nodes[0].inputs)
    bottom_nodes_0_inputs_name_lst = []
    for input in bottom_nodes[0].inputs:
        bottom_nodes_0_inputs_name_lst.append(input.name)

    for i in range(1, len(bottom_nodes)):
        for j in range(len(bottom_nodes[i].inputs)):
            if bottom_nodes[i].inputs[j].name not in bottom_nodes_0_inputs_name_lst:
                bottom_nodes[0].inputs.append(bottom_nodes[i].inputs[j])

    graph.inputs = bottom_nodes[0].inputs
    for i in range(ori_bottom_nodes_0_input_num, len(bottom_nodes[0].inputs)):
        bottom_nodes[0].inputs.pop()

    # graph.inputs[0].shape = [1,
    #       32,
    #       69,
    #       69]
    # graph.inputs[0].dtype = np.float32

    #######
    # 只有从 Conv0 也就是第一个节点开始拆网络时才用得着
    # 指定输入input，去除多余input
    # j = 0
    # for i in range(len(graph.inputs)):
    #     i -= j
    #     if graph.inputs[i].name in model_input_name:
    #         continue
    #     else:
    #         del graph.inputs[i]
    #         j += 1
    ########

    # for i in range(0, len(bottom_nodes), 1):
    #     graph.inputs += bottom_nodes[i].inputs

    top_nodes = []
    for node in graph.nodes:
        for node_t in top_mode_name:
            if node.name == node_t:
                top_nodes.append(node)
        if len(top_nodes) == len(top_mode_name):
            break

    real_graph_outputs = []
    for i in range(0, len(top_nodes), 1):
        top_node = top_nodes[i]
        #############
        # top_node.outputs[0].dtype = np.float32
        #############
        real_graph_outputs += top_node.outputs
    if model_output_name != []:
        for modelOutName in model_output_name:
            for output in graph.outputs:
                if output.name == modelOutName:
                    real_graph_outputs.append(output)

    # 将修改后的graph.outputs变为想要的top_node.outputs
    graph.outputs = real_graph_outputs  # top_node.outputs
    # for i,top_node in enumerate(top_nodes):
    #     top_node.outputs[0].dtype = np.float32

    # 去除多余input
    j = 0
    FinalRemainInputs = []
    for i in range(len(graph.inputs)):
        i -= j
        if (
            graph.inputs[i].name in model_input_name
            and graph.inputs[i].name not in FinalRemainInputs
        ):
            FinalRemainInputs.append(graph.inputs[i].name)
            continue
        else:
            del graph.inputs[i]
            j += 1

    #############
    graph.ir_version = 9
    graph.opset = 7
    #############
    for idx, inShape in enumerate(inputShapes):
        graph.inputs[idx].shape = inShape
    for idx, inType in enumerate(inputTypes):
        graph.inputs[idx].dtype = inType
    for idx, outShape in enumerate(outputShapes):
        graph.outputs[idx].shape = outShape
    for idx, outType in enumerate(outputTypes):
        graph.outputs[idx].dtype = outType
    # self modify
    # graph.inputs[0].shape = [1,1024,64,78]
    # graph.inputs[1].shape = [1,1024,64,78]
    # graph.inputs[0].shape = ["batch_size","channel","height","width"]
    # graph.inputs[1].shape = ["batch_size","channel","height","width"]
    # graph.inputs[2].shape = ["batch_size","channel","height","width"]
    # graph.inputs[0].name = "Tensor_421"
    # graph.inputs[1].name = "Tensor_440"
    # graph.inputs[2].name = "Tensor_441"
    # graph.inputs[3].shape = [1,1,1,60]
    # graph.inputs[4].shape = [1,60,64,64]
    # graph.inputs[0].shape = [10]
    # graph.inputs[0].shape = [1,3,96,544]
    # del graph.inputs[0]
    # graph.inputs[2].inputs[0].outputs[0].dtype = np.uint16
    # graph.inputs[2].inputs[0].outputs[0].shape = [1,1,400,640]
    # graph.inputs[0].dtype=np.float32
    # graph.inputs[1].dtype=np.float32
    # graph.inputs[2].dtype=np.float32
    # graph.inputs[3].dtype=np.float16
    # graph.inputs[4].dtype=np.float16
    # graph.inputs[0].dtype=np.float16
    # graph.inputs[1].dtype=np.float16
    # graph.outputs[1].dtype=np.float32
    # del graph.outputs[1]
    # graph.outputs[0].dtype=np.int64
    # graph.outputs[0].dtype=np.int8
    # graph.outputs[1].dtype=np.float32
    # graph.outputs[1].dtype=np.float32
    # graph.outputs[1].shape = [1,1,1,1]
    #############
    graph.cleanup().toposort()
    #############
    # self modify
    # graph.nodes[0].attrs['inTypes'][-1] = 4
    # del graph.nodes[8].inputs[0]
    # graph.nodes[8].inputs.insert(0, graph.nodes[4].outputs[0])
    # del graph.nodes[6]
    # graph.nodes[4].outputs.append(graph.nodes[5].inputs[0])
    # graph.nodes[0].outputs[0].dtype = np.int64
    # graph.cleanup().toposort()
    #############
    print("====> all nodes num: ", len(graph.nodes))
    onnx.save(gs.export_onnx(graph), save_path)
    reset_ir_version(save_path)


def pad_weight(src_w: np.array, new_oc: int = 0, new_ic: int = 0, bn_var=0.0):
    src_shape = src_w.shape
    new_size = list(copy.deepcopy(src_shape))
    oc = src_shape[0]
    if len(src_shape) > 1:
        ic = src_shape[1]

        if new_oc == 0:
            new_oc = oc
        if new_ic == 0:
            new_ic = ic
        new_size[0] = new_oc
        new_size[1] = new_ic

        new_w = np.zeros(new_size, dtype=np.float32)
        if len(src_shape) == 4:
            new_w[:oc, :ic, :, :] = src_w
        elif len(src_shape) == 3:
            new_w[:oc, :ic, :] = src_w
        elif len(src_shape) == 2:
            new_w[:oc, :ic] = src_w
    else:
        if new_oc:
            new_size[0] = new_oc
            # if not is_bn:
            new_w = np.zeros(new_size, dtype=np.float32) * bn_var
            # else:
            #     new_w = np.ones(new_size, dtype=np.float32)
            new_w[:oc] = src_w
        if new_ic:
            new_size[0] = new_ic
            # if not is_bn:
            new_w = np.zeros(new_size, dtype=np.float32) * bn_var
            # else:
            #     new_w = np.ones(new_size, dtype=np.float32)
            new_w[:oc] = src_w

    return new_w, new_size


def add_dtype(model_path, save_path):
    graph = gs.import_onnx(onnx.load(model_path))
    for node in graph.nodes:
        for output in node.outputs:
            output.dtype = np.float32
        print("kkkkk")
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), save_path)
    reset_ir_version(save_path)
