# 1.三种执行shell 脚本的方式及其不同
执行Shell脚本有多种方式，以下是其中三种主要的方式及其不同：

1. **直接执行脚本文件：**
   - **命令：** `./script.sh`，其中`script.sh`是脚本文件的名称。
   - **说明：** 在终端中直接使用`./`加上脚本文件的名称来执行。这需要确保脚本文件具有执行权限(`chmod +x script.sh`)。
   - **优点：** 直观、简单，适用于简单的脚本。

2. **使用bash命令执行脚本：**
   - **命令：** `bash script.sh` 或 `sh script.sh`，其中`script.sh`是脚本文件的名称。
   - **说明：** 使用`bash`或`sh`命令显式指定解释器来执行脚本。同样，需要确保脚本文件有执行权限。
   - **优点：** 可以通过指定不同的Shell解释器来执行脚本，更灵活。

3. **source命令执行脚本：**
   - **命令：** `source script.sh` 或 `. script.sh`，其中`script.sh`是脚本文件的名称。
   - **说明：** 使用`source`或`.`命令来在当前Shell环境中执行脚本，而不是启动一个新的进程。这样，脚本中的变量和函数可以在当前Shell环境中保持可见。
   - **优点：** 适用于需要在当前Shell环境中设置变量或定义函数的情况。

# 2.shell中对变量的常见操作
在Shell脚本中，变量是用来存储数据的标识符。变量可以包含数字、字符串、文件名等各种类型的数据。以下是Shell中的变量及其操作：

### 1. **定义变量：**
在Shell中，变量的定义不需要指定数据类型，直接使用等号（=）进行赋值。

```bash
variable_name="value"
```

### 2. **访问变量：**
通过在变量名前加美元符号（$）来引用变量的值。

```bash
echo $variable_name
echo ${variable_name}
```

### 3. **特殊变量：**
- **位置参数变量：** `$0` 表示脚本名称，`$1`、`$2` 等表示脚本的参数。
- **特殊字符变量：** `$$` 表示当前进程的ID，`$?` 表示上一个命令的退出状态。

### 4. **字符串操作：**
- **拼接字符串：**
  ```bash
  greeting="Hello"
  name="World"
  result="$greeting, $name!"
  echo $result
  ```

- **获取字符串长度：**
  ```bash
  string="Hello, World!"
  length=${#string}
  echo $length
  ```

- **提取子字符串：**
  ```bash
  string="Hello, World!"
  substring=${string:0:5}  # 从第0个位置开始提取5个字符
  echo $substring
  ```

### 5. **数组：**
在Shell中，可以使用数组来存储多个数值。

```bash
numbers=(1 2 3 4 5)
echo ${numbers[2]}  # 访问数组的第三个元素


echo ${my_array[0]}  # 访问数组的第一个元素
echo ${my_array[1]}  # 访问数组的第二个元素
echo ${my_array[@]}  # 获取数组的所有元素
echo ${#my_array[@]}  # 获取数组的长度


# 遍历元素
for element in "${my_array[@]}"; do
  echo $element
done

```


### 6. **读取用户输入：**
使用`read`命令从用户获取输入，并将输入存储在变量中。

```bash
echo "Enter your name:"
read user_name
echo "Hello, $user_name!"
```

### 7. **变量修改：**
- **重新赋值：**
  ```bash
  count=5
  count=10  # 重新赋值
  ```

- **自增和自减：**
  ```bash
  counter=0
  ((counter++))  # 自增
  ((counter--))  # 自减
  ```

### 8. **删除变量：**
使用`unset`命令可以删除一个变量。

```bash
variable_to_delete="Some value"
unset variable_to_delete
```

这些操作使Shell变得非常灵活，可以用于存储和处理各种类型的数据。根据具体的需求，可以选择合适的变量类型和操作。


# 3.常用运算符

在Shell脚本中，有许多运算符可用于执行不同类型的操作。以下是一些常用的Shell运算符：

### 1. **算术运算符：**
- `+`：加法
- `-`：减法
- `*`：乘法
- `/`：除法
- `%`：取余

```bash
a=5
b=2
echo $((a + b))  # 输出 7
echo $((a - b))  # 输出 3
echo $((a * b))  # 输出 10
echo $((a / b))  # 输出 2
echo $((a % b))  # 输出 1
```

### 2. **关系运算符：**
- `-eq`：等于
- `-ne`：不等于
- `-lt`：小于
- `-le`：小于等于
- `-gt`：大于
- `-ge`：大于等于

```bash
a=5
b=2

if [ $a -eq $b ]; then
  echo "a equals b"
else
  echo "a is not equal to b"
fi
```

### 3. **逻辑运算符：**
- `&&`：逻辑与
- `||`：逻辑或
- `!`：逻辑非

```bash
a=true
b=false

if [ $a == true ] && [ $b == true ]; then
  echo "Both conditions are true"
else
  echo "At least one condition is false"
fi
```

### 4. **赋值运算符：**
- `=`：赋值
- `+=`：追加赋值

```bash
x=10
y=20
let "z = x + y"
echo "z is $z"  # 输出 z is 30

string1="Hello"
string2="World"
concatenated="$string1 $string2"
echo $concatenated  # 输出 Hello World
```

### 5. **字符串运算符：**
- `=`：字符串相等
- `!=`：字符串不相等
- `-z`：字符串为空
- `-n`：字符串非空

```bash
str1="Hello"
str2="World"

if [ "$str1" = "$str2" ]; then
  echo "Strings are equal"
else
  echo "Strings are not equal"
fi
```

这些是一些常见的Shell运算符，它们允许在脚本中执行各种算术、关系、逻辑和字符串操作。根据需要，你可以将它们组合使用。

# 4 shell 中的常见控制结构
在Shell脚本中，有多种控制结构用于控制程序的流程。以下是一些常见的Shell控制结构：
### 1. **条件语句 - if-else：**
```bash
if [ condition ]; then
  # 当条件为真时执行的代码
else
  # 当条件为假时执行的代码
fi
```

### 2. **循环结构 - for 循环：**
```bash
for variable in list; do
  # 循环体中的代码，变量依次取列表中的值
done
```

### 3. **循环结构 - while 循环：**
```bash
while [ condition ]; do
  # 循环体中的代码，当条件为真时执行
done
```

### 4. **循环结构 - until 循环：**
```bash
until [ condition ]; do
  # 循环体中的代码，当条件为假时执行
done
```

### 5. **case 语句：**
```bash
case expression in
  pattern1)
    # 匹配 pattern1 时执行的代码
    ;;
  pattern2)
    # 匹配 pattern2 时执行的代码
    ;;
  *)
    # 默认情况下执行的代码
    ;;
esac
```

### 6. **函数定义和调用：**
```bash
function_name() {
  # 函数体中的代码
}

# 调用函数
function_name
```

### 7. **中断循环 - break 和 continue：**
- `break`：用于跳出循环。
- `continue`：用于跳过循环中的剩余代码，进入下一次循环迭代。

这些控制结构使Shell脚本能够根据不同的条件执行不同的代码块，实现流程的控制和重复执行。你可以根据具体的需求组合使用这些结构来编写灵活的Shell脚本。

# 5 shell中的函数
在Shell脚本中，你可以定义和调用函数以组织和重用代码。以下是Shell中函数的一般结构：

### 1. **函数的定义：**
```bash
function_name() {
  # 函数体中的代码
  # 可以包含任意的Shell命令
  # 也可以接受参数
}
```

或者使用简洁形式：

```bash
function_name() command1; command2; return
```

### 2. **函数的调用：**
```bash
function_name  # 无参数调用

# 或者传递参数
function_name arg1 arg2
```

### 3. **函数参数：**
在函数内部，可以使用`$1`、`$2` 等来获取传递给函数的参数。例如：

```bash
function_with_params() {
  echo "First parameter: $1"
  echo "Second parameter: $2"
}

function_with_params arg1 arg2
```

### 4. **返回值：**
使用 `return` 语句来返回函数的结果。

```bash
function_with_return() {
  result=$(( $1 + $2 ))
  return $result
}

# 调用函数并获取返回值
sum=$(function_with_return 3 4)
echo "Sum is $sum"
```

### 5. **局部变量：**
在函数内部声明的变量默认为局部变量，只在函数内部可见。

### 6. **函数调用时的退出状态：**
函数的退出状态可以通过 `$?` 获取。

```bash
function_with_exit_status() {
  # 函数体中的代码
  return 42
}

function_with_exit_status
echo "Exit status: $?"
```

### 7. **删除函数：**
使用 `unset` 命令可以删除已定义的函数。

```bash
unset function_name
```

这些是Shell脚本中函数的基本结构和用法。使用函数可以提高代码的可读性和可维护性，并允许在脚本中重复使用相同的功能。