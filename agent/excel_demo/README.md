# Excel Agent

## ⚙️ 环境配置

推荐使用 `python >= 3.10` 运行代码。

```bash
# 使用 venv 创建虚拟环境
python3 -m venv excels_test
cd excels_test/bin && source activate && cd -
python3 -m pip install -r requirements.txt
ipython kernel install --name "python3" --user
```

## ▶️ 执行demo代码

需要准备好excel文件，然后执行demo代码。
```bash
model_name=Hunyuan-A13B-Instruct
model_server_ip_addr=http://localhost:port
excel_file=path/to/excel_file
python3 demo.py ${model_name} ${model_server_ip_addr} ${excel_file}
```