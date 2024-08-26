# python 易错知识点 

## reshape

![image-20230915114605387](C:\Users\86155\AppData\Roaming\Typora\typora-user-images\image-20230915114605387.png)



```c++
arr.shape    # (a,b)
arr.reshape(m,-1) #改变维度为m行、d列 （-1表示列数自动计算，d= a*b /m ）
arr.reshape(-1,m) #改变维度为d行、m列 （-1表示行数自动计算，d= a*b /m ）
```

## isinstance 和 type 的区别在于：

- type()不会认为子类是一种父类类型。
- isinstance()会认为子类是一种父类类型。

```python
>>> class A:
...     pass
... 
>>> class B(A):
...     pass
... 
>>> isinstance(A(), A)
True
>>> type(A()) == A 
True
>>> isinstance(B(), A)
True
>>> type(B()) == A
False
```

## == 与is的区别

在Python中，`==`和`is`是两种不同的比较操作符。

1. `==` 操作符用于比较两个对象的值是否相等。它会比较对象的内容或值是否相同。

```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True，因为a和b的内容相同
```

2. `is` 操作符用于检查两个对象是否是同一个对象（即在内存中是否存储了相同的地址）。它比较的是对象的身份。

```python
a = [1, 2, 3]
b = a
print(a is b)  # True，因为a和b指向同一个对象
```

在一般情况下，应该使用`==`来比较两个对象的值是否相等，而使用`is`来检查对象的身份是否相同。但是要注意，对于不可变对象（如整数、字符串等），可能会出现一些意想不到的结果，因为Python可能会对这些对象进行重用，导致`is`的结果为True。因此，对于不可变对象，最好还是使用`==`来进行比较。

## Tuple（元组）

元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号 **()** 里，元素之间用逗号隔开。

元组中的元素类型也可以不相同：

## tuple 相关操作

```python
#!/usr/bin/python3

tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2  )
tinytuple = (123, 'runoob')

print (tuple)             # 输出完整元组
print (tuple[0])          # 输出元组的第一个元素
print (tuple[1:3])        # 输出从第二个元素开始到第三个元素
print (tuple[2:])         # 输出从第三个元素开始的所有元素
print (tinytuple * 2)     # 输出两次元组
print (tuple + tinytuple) # 连接元组

tup1 = ()    # 空元组
tup2 = (20,) # 一个元素，需要在元素后添加逗号
```

## set 相关操作

```python
#!/usr/bin/python3

sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}

print(sites)   # 输出集合，重复的元素被自动去掉

# 成员测试
if 'Runoob' in sites :
    print('Runoob 在集合中')
else :
    print('Runoob 不在集合中')
    
# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')

print(a)

print(a - b)     # a 和 b 的差集

print(a | b)     # a 和 b 的并集

print(a & b)     # a 和 b 的交集

print(a ^ b)     # a 和 b 中不同时存在的元素
```

## dict的几种创建方法

```python
>>> dict([('Runoob', 1), ('Google', 2), ('Taobao', 3)])
{'Runoob': 1, 'Google': 2, 'Taobao': 3}
>>> {x: x**2 for x in (2, 4, 6)}
{2: 4, 4: 16, 6: 36}
>>> dict(Runoob=1, Google=2, Taobao=3)
{'Runoob': 1, 'Google': 2, 'Taobao': 3}
```
## python中的装饰器(decorator)
在Python中，装饰器是一种用于修改函数或类的行为的特殊函数。装饰器允许在`不修改原始函数或类定义`的情况下，向它们`添加功能或修改其行为`。常见的装饰器包括`函数装饰器`和`类装饰器`。

### 1. **函数装饰器**：
函数装饰器是应用于函数的装饰器。它们以被装饰的函数作为参数，并返回一个新的函数或可调用对象，通常用于在调用原始函数之前或之后执行一些操作。

```python
def decorator(func):
    def wrapper(*args, **kwargs):
        # 执行一些额外的操作
        result = func(*args, **kwargs)
        # 执行一些额外的操作
        return result
    return wrapper

@decorator
def some_function():
    # 函数体
    pass
# 调用被装饰后的函数
some_function()
```
### 2. **类装饰器**：
类装饰器是应用于类的装饰器。它们以被装饰的类作为参数，并通常返回一个新的类，可以用来修改或扩展原始类的行为。

```python
class Decorator:
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        # 执行一些额外的操作
        instance = self.cls(*args, **kwargs)
        # 执行一些额外的操作
        return instance

@Decorator
class SomeClass:
    # 类定义
    pass

# 使用被装饰后的类
obj = SomeClass()
```
### 3. 常用的内置装饰器及其用法示例：
#### 1. `@staticmethod`：
   - 用于将方法定义为类的静态方法，不会接收隐式的第一个参数（通常是实例 self）。(不用也可以直接调用)
   示例：
   ```python
   class MyClass:
       @staticmethod
       def static_method():
           print("This is a static method")

   # 可以直接通过类调用静态方法，不需要创建类的实例
   MyClass.static_method()
   ```

#### 2. `@classmethod`：
   - 用于将方法定义为类的方法，第一个参数接收类本身（通常是 cls）。
   示例：
   ```python
   class MyClass:
       class_variable = "Class Variable"

       @classmethod
       def class_method(cls):
           print(cls.class_variable)

   # 可以通过类调用类方法
   MyClass.class_method()
   ```

#### 3. `@property`：
   - 用于将方法转换为属性，可以像访问属性一样访问方法。
   示例：
   ```python
   class MyClass:
       def __init__(self):
           self._x = None

       @property
       def x(self):
           return self._x

       @x.setter
       def x(self, value):
           self._x = value

   obj = MyClass()
   obj.x = 10  # 调用 x.setter 方法
   print(obj.x)  # 调用 x 方法
   ```
#### 4. `@classmethod`与`@staticmethod`的比较：
   - `@classmethod`需要接受类作为第一个参数，而`@staticmethod`不需要接受类或实例作为参数。
   - `@classmethod`通常用于需要操作类属性或调用类方法的情况，而`@staticmethod`适用于不需要类或实例的情况。

这些内置装饰器提供了一种简单而强大的方式来管理类的行为，并且在很多情况下可以使代码更清晰和更易于理解。







# numpy知识点

## numpy 的拼接指令

有两个数组
```python
>>> a
array(［0, 1, 2],
       [3, 4, 5],
       [6, 7, 8］)
>>> b = a*2
>>> b
array(［ 0, 2, 4],
       [ 6, 8, 10],
       [12, 14, 16］)
```

##### 1、水平组合

```python
>>> np.hstack((a,b))
array(［ 0, 1, 2, 0, 2, 4],
       [ 3, 4, 5, 6, 8, 10],
       [ 6, 7, 8, 12, 14, 16］)

>>> np.concatenate((a,b),axis=1)
array(［ 0, 1, 2, 0, 2, 4],
       [ 3, 4, 5, 6, 8, 10],
       [ 6, 7, 8, 12, 14, 16］)
```
##### 2、垂直组合
```python
>>> np.vstack((a,b))
array(［ 0, 1, 2],
       [ 3, 4, 5],
       [ 6, 7, 8],
       [ 0, 2, 4],
       [ 6, 8, 10],
       [12, 14, 16］)

>>> np.concatenate((a,b),axis=0)
array(［ 0, 1, 2],
       [ 3, 4, 5],
       [ 6, 7, 8],
       [ 0, 2, 4],
       [ 6, 8, 10],
       [12, 14, 16］)
```

##  numpy.cumsum()

> numpy.cumsum(a, axis=None, dtype=None, out=None)  
> axis=0，按照行累加。  
> axis=1，按照列累加。  
> axis不给定具体值，就把numpy数组当成一个一维数组。

>>> a = np.array([[1,2,3], [4,5,6]])

>>> a

```css
    array([[1, 2, 3],

          [4, 5, 6]])
```

>>> np.cumsum(a)

```cpp
    array([ 1,  3,  6, 10, 15, 21])
```

>

>>> np.cumsum(a,axis=0) #按照**行**累加，行求和

```ruby
array([[1, 2, 3],

       [5, 7, 9]])    
  
                        [1, 2, 3]------>     |1     |2     |3    |

                        [4, 5, 6]------>     |5=1+4 |7=2+5 |9=3+6|  
```

>>> np.cumsum(a,axis=1) **列**累加，列求和

```ruby
array([[ 1,  3,  6],

       [ 4,  9, 15]])
                        [1, 2, 3]------>     |1     |2+1    |3+2+1   |

                        [4, 5, 6]------>     |4     |4+5    |4+5+6   |  
```

>>> np.cumsum(a, dtype=float) # 指定输出类型。

__注意啦！没有指定_轴参数(axis)_！输出就变成1维数组了。

```cpp
  array([  1.,  3.,  6.,  10.,  15.,  21.])第一步:每个值都变成float了

  array([1，1+2=3，1+2+3=6，1+2+3+4=10，1+2+3+4+5=15，1+2+3+4+5+6=21]）第二部：累加
```

## python 中正则表达式的使用

在Python中，`re`模块提供了一系列用于处理正则表达式的方法。以下是一些常用的方法：

1. **`re.compile(pattern, flags=0)`：**

   - 用于编译正则表达式，返回一个正则表达式对象。可以提高正则表达式的执行效率。

   ```python
   import re
   pattern = re.compile(r'\d+')
   ```

2. **`re.match(pattern, string, flags=0)`：**

   - 尝试从字符串的开头匹配正则表达式。如果匹配成功，返回一个匹配对象，否则返回`None`。

   ```python
   result = re.match(r'\d+', '123abc')
   ```

3. **`re.search(pattern, string, flags=0)`：**

   - 在字符串中搜索匹配正则表达式的第一个位置。如果匹配成功，返回一个匹配对象，否则返回`None`。

   ```python
   result = re.search(r'\d+', 'abc123def')
   ```

4. **`re.findall(pattern, string, flags=0)`：**

   - 返回字符串中所有与正则表达式匹配的非重叠的子串，以列表形式返回。

   ```python
   result = re.findall(r'\d+', 'abc123def456')
   ```

5. **`re.finditer(pattern, string, flags=0)`：**

   - 返回一个迭代器，生成匹配正则表达式的所有非重叠子串的匹配对象。

   ```python
   result = re.finditer(r'\d+', 'abc123def456')
   ```

6. **`re.sub(pattern, repl, string, count=0, flags=0)`：**

   - 用新的字符串替换匹配正则表达式的子串。可选参数`count`用于指定替换次数。

   ```python
   new_string = re.sub(r'\d+', 'X', 'abc123def456')
   ```

7. **`re.split(pattern, string, maxsplit=0, flags=0)`：**

   - 使用正则表达式匹配的模式分割字符串，返回分割后的子串列表。可选参数`maxsplit`用于指定最大分割次数。

   ```python
   parts = re.split(r'\d+', 'abc123def456')
   ```

这些方法提供了灵活的正则表达式处理工具，你可以根据具体的需求选择适当的方法。同时，可以通过使用正则表达式的标志（flags）来修改匹配的行为，例如，`re.IGNORECASE`用于忽略大小写。

## 字符串前的 r 的作用

在Python中，字符串前面的 `r` 表示“原始字符串”（raw string）。当你在字符串前面加上 `r` 前缀时，意味着该字符串中的反斜杠 `\` 将被视为普通字符而非转义字符。这对于处理正则表达式、文件路径等包含反斜杠的字符串非常有用。

例如，考虑以下两个字符串：

```python
normal_string = "This is a line break:\nSecond line."
raw_string = r"This is a line break:\nSecond line."

>>> print('\n')       # 输出空行

>>> print(r'\n')      # 输出 \n
\n
```

在 `normal_string` 中，`\n` 被解释为一个换行符，而在 `raw_string` 中，`\n` 被当作两个字符（反斜杠和字母 'n'）对待。使用原始字符串可以在不使用双反斜杠的情况下表示反斜杠，这对于正则表达式等情况非常方便。

```python
import re

pattern_regular = re.compile("\\d+")
pattern_raw = re.compile(r"\d+")
```

在上述正则表达式的例子中，`pattern_regular` 需要使用 `\\d+` 表示匹配数字，而 `pattern_raw` 可以使用 `r"\d+"`，避免了双反斜杠的使用，使正则表达式更加清晰。

## 多行语句

Python 通常是一行写完一条语句，但如果语句很长，我们可以使用反斜杠 **\** 来实现多行语句，例如：

```python
total = item_one + \
        item_two + \
        item_three
```







# matplotlib 常用指令

## subplot
1. 常用的调用方式
最常用的就是传入关键字三个数字，可以省略逗号，当省略逗号的时候figure中最多只能显示9个坐标轴(多于9个缩写就不能区分了，比如3412，不知道是要创建一个3行4列第12个还是3行41列第2个)，而有逗号分隔则没有数量限制。

还有一点在jupyter notebook中，当你调用plt.subplot(3, 4, 6)的时候，由于最后一个图的右边和下面都没有坐标系显示了，所以直接没画，实际仍然是

# pandas易错点

 ## Series如何判断行索引是否包含该行索引
```python
if 1 in df.index.values:
    print(df.loc[1])
else:
    print('None')
```

