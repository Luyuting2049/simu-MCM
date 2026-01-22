**如果是VScode打开的话记得ctrl+shift+v打开预览模式看这个README**

## 结构说明
#### PDESolver
    （偏微分方程）求解器，封装进类了，主函数里直接调用就好了  
#### C-t_Image
    里面放的浓度-时间变化图  
        浓度-时间（min）  
        浓度-时间（h）  
        浓度（距离km，时间h）（三维图）  
        test1 test4就是画图用的  
#### SensTest_Image
    两张灵敏度测试图（有点简单，写的时候看情况画需要的图）  
        VaryU 固定D在基准值50，改变水流速度看取水口浓度变化情况  
        VaryD 固定U为2，改变弥散系数看取水口浓度情况  
#### config.yaml
    配置文件（官方给的直接放进来的）  
#### main.py
    主程序，创立了config类，以及用来循环不同U和D还有画图的  
        （其他模型也用这个的话我到时候就把config单独拆个程序）  

## 污染物模型思路
MOL+ODE求解器
ODE求解器使用：https://docs.scipy.org.cn/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
详见PDESolver，主要就是30km切成N段（N在config里设置）  
然后每个点对浓度对x的偏导进行近似，带入后变成一组常微分方程  
再用现成的ODE求解器来求解  

## 参数配置
参数配置统一放在config里面了（后续修改的话只用修改config）,用的是官方给的yaml文件
需安装pyyaml库，用的时候import yaml但依赖都统一放在requirement.txt里了直接下载就行
(防止需要自己加参数不知道语法)
### YAML基础语法速查
1. **键值对**：`key: value`（冒号后必须有空格）
2. **缩进**：用2个空格表示层级（不要用Tab），如：
   ```yaml
   parent:
     child: value
   ```
3. **列表**：用短横线表示，如：
   ```yaml
   fruits:
     - apple
     - banana
   ```
4. **行内写法**：`list: [a, b, c]` 或 `dict: {key: value}`
5. **多行文本**：
   - 保留换行：`text: |`（竖线后换行）
   - 折叠一行：`text: >`（>后换行）
6. **数据类型**：自动识别字符串、数字（`age: 20`）、布尔值（`flag: true`）、空值（`empty: null`）
7. **注释**：以`#`开头，如：`# 这是注释`
8. **特殊字符**：包含空格或符号的键需要引号，如：`"first name": 张三`

**示例**：
```yaml
# 配置文件示例
database:
  host: "localhost"
  port: 3306
  users: [admin, guest]
  settings:
    timeout: 30
    retry: true
  description: |
    这是多行
    文本描述
```

