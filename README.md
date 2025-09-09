<div align="center">
<h1>SEAL</h1>
<h3>
SEAL: Structure and Element Aware Learning Improves Long Structured Document Retrieval
<h3>
</div>
  
This is the official repository for our paper, **"SEAL: Structure and Element Aware Learning Improves Long Structured Document Retrieval"**.
  
<p align="center">
<img width="100%" alt="image" src="overview.jpg">    
</p>

## 🚀 Framework Overview

The core idea of SEAL is to enable the model to learn both the **structural information** and **element-level semantics** of documents during the fine-tuning process. This is achieved through two synergistic contrastive learning tasks:

1.  **Structure-Aware Learning (SAL)**
    
    *   **Goal**: To teach the model to understand the macro-level structural hierarchy of a document (e.g., the relationship between headings and paragraphs).
        
    *   **Implementation**: To achieve this, we form a positive pair consisting of the original document (with HTML tags) and a "plain text" version of the same document (with all tags removed). This strategy compels the model to recognize that the semantic essence and structure of a document remain consistent, whether the structural cues (HTML tags) are explicit or implicit.
        
    *   **In the code**: The `_remove_tag` function in `dataset.py` is responsible for creating this plain text version (referred to as `passages_untagged`). The SAL loss (`q_p_untagged_loss`) is then computed via contrastive learning between the query embedding (`q_dense_vecs`) and the embedding of this tagless document (`p_untagged_dense_vecs`), as seen in `modeling.py`.
        

2.  **Element-Aware Alignment (EAL)**
    
    *   **Goal**: To enhance the model's ability to discriminate fine-grained elements within a document (e.g., text within a `<p>` tag).
        
    *   **Implementation**: We introduce a masking strategy where a random subset of HTML elements and their content are removed from the document. The model is then tasked to associate the query with this partially "damaged" document. This forces the model to develop a deeper understanding of the remaining elements, as it cannot rely on the full document context.
        
    *   **In the code**: This is handled by the `_mask_element` function in `dataset.py`, which produces the `passages_masked` variant. The corresponding EAL loss (`q_p_masked_loss`) is calculated between the query and the embedding of the masked document (`p_masked_dense_vecs`) in `modeling.py`.
        

Through the joint optimization of these two tasks, SEAL learns a unified embedding space where document representations contain both structural information and fine-grained semantic understanding, significantly improving retrieval performance on long structured documents.

## 💡 Datasets
We introduce **StructDocRetrieval**, a dataset designed for training and evaluating long structured document retrieval models.

### Statistic
| Split       | Query   | Doc.    | Avg. Words Query | Avg. Words Doc. |
|-------------|---------|---------|------------------|-----------------|
| **Train**   | 23,816  | 23,816  | 12.82            | 10,849          |
| **Test**    | 3,404   | 3,404   | 13.04            | 10,535          |
| **Evaluation** | 6,804  | 6,804   | 12.74            | 11,047          |

*Note:* The "Avg. Words" represents the average number of words in queries and documents.


### Data Format

The data is provided in a `.json` file, where each line is a JSON object with the following fields:

*   `query` (str): The user query.
    
*   `content` (str): The full document content, including HTML tags.
    

### Example

Here is an example from `train.json` for illustration purposes.

```Json
{
    "query": "VS Code配置Python环境",
    "content": "<body><link/><link/><div><div><main><div><div><div><div><h1>【保姆级教程】VS Code安装配置Python、Jupyter、C</h1></div><div><div><div><span>已于 2025-03-21 16:15:51 修改</span><div><span>阅读量6.8k</span><div><span>点赞数 39 </span></div></div></div></div><div><div><span>文章标签：</span></div></div><div><span>于 2025-03-11 11:28:27 首次发布</span></div><div><div><div> 版权声明：本文为博主原创文章，遵循版权协议，转载请附上原文出处链接和本声明。 </div><div> 本文链接：</div></div></div></div></div></div><article><div><div><h2>一、下载安装VS Code</h2><blockquote><p><br/> (官网可免费下载，下载速度不会慢)</p></blockquote><p>选择适合自己电脑的版本，这里我选的是Windows，点击即开始下载。</p><p>下载完成后双击安装，后面是一系列傻瓜操作，直至安装完成。</p><p><strong>需要注意的地方：</strong><br/> 1、安装路径。推荐不要装在C盘，可装在D盘；<br/> 2、选择附加任务，可以全部勾上，后续就不用再额外添加PATH了。</p><p>以上安装完成后，在扩展中搜索 “chinese” ，安装中文语言包，安装完成后会自动重启。</p><h2>二、VS Code配置Python环境</h2><p>如果电脑已经下载了python，那么可以省略下载这一步，后面直接配置环境即可。</p><h3>1、下载Python</h3><p>根据需求选择下载方式，这里我选择下载miniconda。</p><blockquote><p><strong>安装内容</strong>：<br/><strong>Anaconda</strong>：是一个功能强大的 Python 发行版，包含了 Python 解释器以及大量常用的科学计算、数据分析、机器学习等相关的库和工具，比如 NumPy、Pandas、Matplotlib等，还自带了包管理工具和环境管理工具。安装包较大，通常在几百 MB 到 1GB 左右。</p><p><strong>Miniconda</strong>：是 Anaconda 的轻量级版本，只包含了 Python 解释器和最基本的包管理工具等核心组件，不包含大量预安装的第三方库。安装包相对较小，一般在几十 MB 左右。</p><p><strong>直接安装 Python</strong>：只安装了 Python 解释器本身，没有附带任何额外的第三方库或工具。如果需要使用特定的库，需要用户手动逐个安装。</p></blockquote><blockquote><p><strong>适用场景</strong>：<br/><strong>Anaconda</strong>：适合从事数据科学、机器学习、数据分析等领域的专业人员，以及需要在多个项目中使用大量不同库的开发者。它可以方便地管理不同项目所需的各种环境和库，减少环境配置的麻烦。<br/><strong>Miniconda</strong>：适用于那些对 Python 环境有一定定制需求，不想一开始就安装大量不必要库的用户。例如，有一定 Python 基础，知道自己具体需要哪些库，希望按需安装的开发者。<br/><strong>直接安装 Python</strong>：适合对 Python 环境要求非常简单，只需要使用 Python 基本功能，或者对系统资源占用要求极高，不希望有任何多余组件的用户。比如，只需要用 Python<br/> 编写一些简单的脚本，或者在资源有限的嵌入式设备上使用 Python 的情况。</p></blockquote><h4>（1）下载miniconda安装包</h4><p>1.打开清华镜像源网站https://mirrors.tuna.tsinghua.edu.cn/（说明：miniconda在清华镜像源下载速度比在国外服务器下载快），找到anaconda后点击。<br/></p><h4>（2）安装miniconda程序</h4><p>点击运行上一步下载好的.exe文件，点击Next，接下来就是一直顺着提示操作。</p><p><strong>需要注意的是</strong>：</p><p>（1）建议更改安装位置；</p><p>（2）勾选添加环境（后续不用再做添加环境的操作）。</p><h4>（3）打开cmd命令行，检验python环境是否安装好</h4><pre><code>where python </code></pre><h3>2、VSCode中配置Python</h3><h4>（1）安装第三方包：flake8、yapf</h4><p>flake8用于检查代码规范和语法错误。</p><p>yapf则是一个代码格式化工具，可以帮助我们美化代码。</p><p>输入以下命令安装flake8：</p><pre><code>pip install flake8 </code></pre><p>输入以下命令安装yapf：</p><pre><code>pip install yapf </code></pre><h4>（2）在vscode扩展中搜索python插件并安装</h4><h4>（3）创建本地文件夹作为python项目文件夹，并配置工作区域。</h4><p>先创建好文件夹，随后在vscode中选择该文件夹并打开。</p><p>根据以下步骤，打开设置，<strong>并将以下代码内容放入。</strong></p><pre><code> \"python.linting.flake8Enabled\": true, \"python.formatting.provider\": \"yapf\", \"python.linting.flake8Args\": [\"--max-line-length=248\"], \"python.linting.pylintEnabled\": false, \"notebook.lineNumbers\": \"on\" </code></pre><h4>（4）编写python文件</h4><p>创建一个python文件，并编写代码，选择自己的python解释器运行。</p><pre><code>print(\"Hello VSCode\") </code></pre><h2>三、VS Code配置Jupyter</h2><h3>1、安装jupyter扩展</h3><h3>2、pip安装jupyter</h3><p>在电脑搜索栏输入“cmd”打开命令行，输入以下命令安装jupyter。</p><pre><code>pip install jupyter -i https://pypi.tuna.tsinghua.edu.cn/simple/ </code></pre><h3>3、运行ipynb文件</h3><p>同理上方。</p><h2>四、VS Code配置C语言环境</h2><h3>1、下载安装GCC编译器</h3><p>这里分享我找到的资源，可以直接下载，或查找其他资源。<br/> 版本：x86 64-13.2.0-release-win32-seh-msvcrt-rt v11-rev1<br/> 链接：<br/> 提取码：8888</p><p>将下载好的文件解压在一个地方，确保不会乱动，这里我选择放在D盘（自由选择，要保证后期不会乱动）<br/></p><h3>2、配置编译环境</h3><p>（1）复制编译器文件夹中bin文件夹的地址（我的是D:\mingw64\bin）</p><p>（2）任务栏搜索：环境变量，并打开，将刚才复制的路径按照以下操作放入对应的位置，<strong>随后点击确认</strong>。</p><p>（3）检验是否配置成功</p><p>在搜索栏输入cmd打开命令端，运行 <strong>gcc -v</strong> 这条命令（注意 - 前有个空格），最后一行显示gcc编译器版本即配置成功。<br/><strong>ps</strong>：如果前面步骤均正确，但输入命令后显示 “gcc不是内部或外部命令” ，可以尝试重启电脑再检验。</p><h3>3、运行C语言程序</h3></div></div></article></div><div><div><div> 确定要放弃本次机会？ </div><span>福利倒计时</span><div><i>:</i><i>:</i></div><div><p><span>立减 ¥</span></p><span>普通VIP年卡可用</span></div></div></div><div><div><div><ul><li><div><span>点赞</span></div></li><li><div><span>踩</span></div></li><li><div><div><span> 收藏 </span></div></div><div><div> 觉得还不错? <span> 一键收藏 </span></div></div></li><li><div><button>知道了</button></div><div><span>评论</span></div></li><li><div><div><div><div>扫一扫 </div></div></div></div></li></ul></div></div></div><div><div><div><div><div><div><span>12-19</span><span> 1万+ </span></div></div></div></div></div></div><div><div><span>参与评论</span><span>您还未登录，请先</span><span>登录</span><span>后发表或查看评论</span></div></div><div><div><div><div><div><div><span>03-08</span><span> 1428 </span></div></div></div></div></div><div><div><div><div><div><span>01-26</span><span> 1万+ </span></div></div></div></div></div><div><div><div><div><div><span>05-10</span><span> 1575 </span></div></div></div></div></div><div><div><div><div><div><span>12-17</span><span> 4035 </span></div></div></div></div></div><div><div><div><div><div><span>03-25</span><span> 4083 </span></div></div></div></div></div><div><div><div><div><div><span>08-01</span><span> 2004 </span></div></div></div></div></div><div><div><div><div><div><span>07-14</span><span> 27万+ </span></div></div></div></div></div><div><div><div><div><div><span>07-29</span><span> 246 </span></div></div></div></div></div><div><div><div><div><div><span>10-24</span><span> 1951 </span></div></div></div></div></div><div><div><div><div><div><span>03-08</span><span> 5236 </span></div></div></div></div></div><div><div><div><div><div><span>03-20</span><span> 1129 </span></div></div></div></div></div><div><div><div><div><div><span>11-10</span><span> 899 </span></div></div></div></div></div><div><div><div><div><div><span>05-06</span><span> 3585 </span></div></div></div></div></div><div><div><div><div><div><span>11-15</span><span> 1492 </span></div></div></div></div></div><div><div><div><div><div><span>12-04</span><span> 4336 </span></div></div></div></div></div><div><div><div><div><div><span>12-25</span><span> 1363 </span></div></div></div></div></div><div><div><div><div><div><span>10-19</span><span> 2万+ </span></div></div></div></div></div><div><div><div><div><div><span>09-13</span></div></div></div></div></div></div></main><aside><div><div><div><div><p><span> 博客等级 </span></p><span>码龄5年</span></div></div></div><div><dl><dd>110</dd><dt>点赞</dt></dl><dl><dd>307</dd><dt>收藏</dt></dl><dl><dd><span>56</span></dd><dt>粉丝</dt></dl></div></div><div><h3>热门文章</h3></div><div><h3>分类专栏</h3><div><ul><li><span>2篇</span></li><li><span>1篇</span></li><li><span>2篇</span></li><li><span>2篇</span></li><li><span>2篇</span></li></ul></div></div><div><h3>最新评论</h3><div><ul><li><p><span>刚分享那会儿是免费的，现在它升级了变成积分获取了，暂时没发现哪里能改噢</span></p></li><li><p><span>数据集下载需要积分，无法免费获取</span></p></li></ul></div></div><div><h3>最新文章</h3></div><div><div><h3>目录</h3></div></div></aside></div><div><aside><div><div><div><h3>目录</h3></div></div><div><h3>分类专栏</h3><div><div><ul><li><span>2篇</span></li><li><span>1篇</span></li><li><span>2篇</span></li><li><span>2篇</span></li><li><span>2篇</span></li></ul></div></div></div></div></aside></div><div><aside><div><div><div><h3>目录</h3></div></div></div></aside></div></div><div><div><div><span>评论</span></div><div><div>被折叠的 条评论 </div></div></div><div><div><div> 添加红包 </div><form><div><label>祝福语</label><p>请填写红包祝福语或标题</p></div><div><label>红包数量</label><div><input/><span>个</span></div><p>红包个数最小为10个</p></div><div><label>红包总金额</label><div><input/><span>元</span></div><p>红包金额最低5元</p></div><div><label>余额支付</label><div> 当前余额<span>3.43</span>元 </div></div><div><div> 需支付：<span>10.00</span>元 </div><button>取消</button><button>确定</button></div></form></div></div><div><div><button>下一步</button></div><div><button>知道了</button></div></div></div><div><div><div><div><footer><div> 领取后你会自动成为博主和红包主的粉丝 </div></footer></div><div><div><header><div><div>hope_wisdom</div> 发出的红包 </div></header></div></div></div></div></div><div><div>实付<span>元</span></div><div><div><div><span>点击重新获取</span></div></div><div><span>扫码支付</span></div></div><div><input/><span>钱包余额</span><span>0</span><div><div><div><p>抵扣说明：</p><p> 1.余额是钱包充值的虚拟货币，按照1:1的比例进行支付金额的抵扣。<br/> 2.余额无法直接购买下载，可以购买VIP、付费专栏及课程。</p></div></div></div></div></div></body>"
}

```

## 📜 Citation

If you find our work useful, please consider citing our paper.

```python
@inproceedings{seal,
  author       = {Xinhao Huang and
                  Zhibo Ren and
                  Yipeng Yu and
                  Ying Zhou and
                  Zulong Chen and
                  Zeyi Wen},
  title        = {SEAL: Structure and Element Aware Learning Improves Long Structured Document Retrieval},
  booktitle    = {{EMNLP}},
  publisher    = {Association for Computational Linguistics},
  year         = {2025}
}
```

## 🙏 Acknowledgements

Our implementation is based on the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) project. We thank them for their excellent work.

