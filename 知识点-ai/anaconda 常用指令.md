# anaconda 常用指令

## 环境管理相关

### 查看当前所有的环境

```shell
conda env list
```

### 查看conda环境管理命令帮助信息
```text
conda create --help
```
创建出来的虚拟环境所在的位置为conda路径下的env/文件下,,默认创建和当前python版本一致的环境.
```text
conda create --name envname
```
### 创建环境
创建新环境时指定python版本为3.6，环境名称为python36

```text
conda create --name python36 python=3.6
conda create --name BigRec python=3.9

conda create --name TALLRec python=3.10
```

切换到环境名为python36的环境（默认是base环境），切换后可通过python -V查看是否切换成功

```text
conda activate python36
```

返回前一个python环境

```text
conda deactivate
```

### 显示已创建的环境，会列出所有的环境名和对应路径

```text
conda info -e
```

### 删除虚拟环境

```text
conda remove --name envname --all
```

指定python版本,以及多个包
```text
conda create -n envname python=3.4 scipy=0.15.0 astroib numpy
```
### 查看当前环境安装的包
```text
conda list   ##获取当前环境中已安装的包
conda list -n python36   ##获取指定环境中已安装的包
```

### 克隆一个环境

```text
# clone_env 代指克隆得到的新环境的名称
# envname 代指被克隆的环境的名称
conda create --name clone_env --clone envname

conda create --name recbole_clone --clone recbole
#查看conda环境信息
conda info --envs
```

### 构建相同的conda环境(不通过克隆的方法)

```text
# 查看包信息
conda list --explicit

# 导出包信息到当前目录, spec-file.txt为导出文件名称,可以自行修改名称
conda list --explicit > spec-file.txt

# 使用包信息文件建立和之前相同的环境
conda create --name newenv --file spec-file.txt

# 使用包信息文件向一个已经存在的环境中安装指定包
conda install --name newenv --file spec-file.txt
```

查找包

```text
#模糊查找，即模糊匹配，只要含py字符串的包名就能匹配到
conda search py   

##查找包，--full-name表示精确查找，即完全匹配名为python的包
conda search --full-name python
```



## recbole 配置环境

```
1. 装torch 按照官网配置
2. conda install -c aibox recbole


3. pip install ray
4. conda install texttable
5. pip install thop

pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu{1}/torch_stable.html
```

## 新环境配置到jupyter notebook 中
```
conda activate TextMining #TextMining为环境名
pip install ipykernel 
python -m ipykernel install --name TextMining #TextMining为环境名

conda activate TextMinin #TextMining为环境名
pip install ipykernel 
python -m ipykernel install --name py310 #TextMining为环境名
```


## 根据文件创建环境

```bash
pip install -r requirements.txt
```