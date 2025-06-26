# ResNet18模型-转换推理指导
## 概述
ResNet（Residual Network，残差网络）在2015年的ImageNet大规模视觉识别挑战赛（ILSVRC）中获得了图像分类和物体识别的优胜。ResNet通过其独特的残差块设计和高效的训练方法，显著提升了模型的性能。
残差块由两部分组成：一个主路径和一个或多个卷积层，以及一个短路连接。主路径负责学习输入到输出的映射，而短路连接则将输入直接连接到输出。这种设计确保了信息可以直接传递，避免了在深层网络中信息的丢失和梯度消失的问题‌。
ResNet的不同版本（如ResNet18、ResNet50、ResNet101等）在各种计算机视觉任务中表现出色，成为深度学习领域的重要里程碑‌。
### 1.1 论文地址
[ResNet18论文](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fpdf%2F1512.03385.pdf)
### 1.2 代码地址
[ResNet18代码](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fpdf%2F1512.03385.pdf)
## 模型准备
可以通过pytorch的pre-trained模型仓库获取训练好的pth模型(不同的torch版本获取的模型有差异)，然后导出onnx模型，参考如下方式
```python
    model = torchvision.models.resnet18(weights=
        torchvision.models.resnet.ResNet18_Weights.DEFAULT)
    model.eval()
    dummy_input = torch.rand(1,3,224,224)
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model,
                     dummy_input, 
                     output_file_path,
                     export_params=True,
                     opset_version=11, 
                     do_constant_folding=True, 
                     input_names=input_names, 
                     output_names=output_names)
```

### 2.1 模型优化（可选）
直接export出来的模型可能存在冗余的、未优化折叠的算子或结构，可以通过onnxsim库进行优化。
```python
onnx_model = onnx.load('resnet_18.onnx')
model_simp, check = onnxsim.simplify(onnx_model, test_input_shapes={'input': [1,3,224,224]})
onnx.save(model_simp, 'resnet_18_sim.onnx')
```

## 模型推理
### 3.1 模型转换推理
获取release版本的[nextncc包](www.baidu.com)(链接更新)（yaml配置文件使用说明文档，参数说明？）

解压文件
```
cd nextncc-windows-dev
Nvpncc.exe --taskConfigPath=config\\TaskConfig.yaml
```
参数说明：
- taskConfigPath： （配置文件说明文档？？？这个才是大头）

--------支持转换，手动infer-tool运行、转换+运行自动比较相似度，这个怎么定---------


运行成功后会在配置的output目录下，infer/infer_out/model name/下保存运行后板子的推理输出的二进制文件。
同时会输出板端推理结果和onnxruntime输出结果的相似度，以便验证板子输出结果的正确性。
```
layer: model_name_1.bin<=>tensor_name.bin
cpp range [-124.0, 18.0]
ref range [-124.0, 18.0]
cos sim: 0.9999663829803467 min:0.0 max:4.0
```

