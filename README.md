# FireRedASR-AED-ONNX

本项目提供了 [FireRedASR-AED](https://github.com/FireRedTeam/FireRedASR) torch 模型转 onnx 模型的脚本，具有以下特性：
- 导出的 onnx 模型支持 beam search
- 适配了 wenet 微调后的 FireRedASR-AED 模型


## 环境准备
克隆 [FireRedASR](https://github.com/FireRedTeam/FireRedASR) 项目，根据项目的 README 配置程序运行环境。在此基础上，安装 onnx、onnxruntime 包。

将本项目下的 python 程序拷贝到 FireRedASR 项目下的 `example` 目录下。


## 使用说明
torch 模型转 onnx 模型。
```bash
model="FireRedASR-AED-L/model.pth.tar"
encoder="onnx_encoder"
decoder="onnx_decoder"
encoder_int8="onnx_encoder_int8"
decoder_int8="onnx_decoder_int8"
cmvn="FireRedASR-AED-L/cmvn.ark"

python to_onnx.py \
--model $model \
--encoder $encoder \
--decoder $decoder \
--encoder_int8 $encoder_int8 \
--decoder_int8 $decoder_int8 \
--cmvn $cmvn
```

测试导出的 onnx 模型
```bash
# wavs 为待转写的音频目录
find wavs -type f -name "*.wav" > wavlist.txt

encoder_path="onnx_encoder_int8/encoder_int8.onnx"
decoder_path="onnx_decoder_int8/decoder_int8.onnx"
cmvn_path="FireRedASR-AED-L/cmvn.ark"
dict_path="FireRedASR-AED-L/dict.txt"
spm_model="FireRedASR-AED-L/train_bpe1000.model"
wavlist="wavlist.txt"
hypo="hypos.txt"
beam_size=3

python test_onnx_model.py \
--encoder $encoder_path \
--decoder $decoder_path \
--cmvn $cmvn_path \
--dict $dict_path \
--spm_model $spm_model \
--wavlist $wavlist \
--hypo $hypo \
--beam_size $beam_size \
--provider "CPUExecutionProvider"
```

使用 wenet 微调后，使用 `convert_wenet_ckpt_to_FireRedAED.py` 脚本将微调后的模型进行转换，然后就可以使用 `to-onnx.py` 将微调后的模型导出为 onnx 格式。
```bash
python --wenet_model wenet_model --fireredaed_model fireredaed_model
```


## 参考
- [wenet](https://github.com/wenet-e2e/wenet/tree/main/wenet/models/firered)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/whisper)
