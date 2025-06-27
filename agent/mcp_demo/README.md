# MCP Agent

## ⚙️ 环境配置

1. 安装node.js,npm,npx
```bash
yum install -y nodejs npm
# 检查npm安装是否成功
npm -v
npm install -g npx
npx -v
```
安装后其中 `node.js >= v20.7.0` `npm >= 10.1.0`，如果`node.js`版本为18，会存在异常。

如果无法安装`node.js`，推荐使用`docker`启动MCP服务，可参考开源项目 [supergateway](https://github.com/supercorp-ai/supergateway)

2. 推荐使用 `python >= 3.10` 运行代码。

```bash
# 使用 venv 创建虚拟环境
python3 -m venv mcp
cd mcp/bin && source activate && cd -
python3 -m pip install -r requirements.txt
```

## ▶️ 执行demo代码

执行demo代码。
```bash
model_name=Hunyuan-A13B-Instruct
model_server_ip_addr=http://localhost:port
python3 demo.py ${model_name} ${model_server_ip_addr}
```

## ⚠️ 风险提示

用户在创建 MCP Server 时可能存在命令注入（Command Injection）及间接提示注入（Indirect Prompt Injection）风险，请谨慎处理工具调用逻辑并采取适当的安全防护措施。