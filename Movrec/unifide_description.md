###### 工作流程

***unifide***和crawl\_doubao\_me有一样的输入，即用户的豆瓣id；和Doumini\_V2有同样的输出。这个程序工作流程简略如下：

 	首先用户输入豆瓣id，程序调用crawl\_doubao\_me对用户主页进行爬取；

 	在爬取得到的数据中将<out\_prefix>\_movies\_wish.csv 文件（想看的电影）的title一栏所有数据存储在一个名叫wish\_list的列表（列表大小为100。若title一栏数据超过100，则取前100个数据）中；

 	调用crawl\_doubao\_me，将wish\_list赋给Doumini\_V2中的movie\_names；最终Doumini\_V2的输出就是程序的输出

##### 注意

**需要预先运行 Doumini\_V2 并保存相似度矩阵** np.save('similarity\_matrix.npy', similarity)

##### 问题

1. 在找原本三个程序的io时，发现crawl\_doubao\_me程序的输出内容过多，有用户个人数据统计（JSON格式总文件（<out\_prefix>.json））、电影统计、广播统计、书籍统计。目前只用到了电影统计，并且只需要电影统计表格中的电影名。
2. 爬取的表格中titel一列中一部电影有多个名字（如a/AD/Ad/dd，不同名字用/分隔）,且这多个名字常常语言不同且十分广泛。目前unfide将title一列所有数据都传入列表movie\_names中，不确定是否有影响。
3. movie\_names中只有想看电影没有看过和收藏电影。
4. 在查看总数据库movie.cvs时 发现由于格间距问题，一些数据无法正常显示（如id号变为xxxxx），感觉这应该没有什么影响。

##### 方法

1. 认为用户个人数据统计可以用来作为输出（类似于qq音乐的年终统计）
2. 过量爬取问题认为可以修改爬虫或在unifide中增加函数，目前已经设计函数用于提取movies\_wish.csv中的title列，可以再设计函数只提取每部电影第一个名字（中文名）（此时wish\_list与Doumini\_V2原输入完全一致）
3. movie\_names中只有想看电影 认为可以全部加在一起，或者设置一种权重机制
