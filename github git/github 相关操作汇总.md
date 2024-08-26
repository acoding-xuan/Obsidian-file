https://www.bilibili.com/video/BV1HM411377j/?p=6&spm_id_from=pageDriver&vd_source=1a17967c9b5ad16c1e6d4d30a40550ab
## 将远程仓库克隆到本地
1. 现在远程创建仓库，然后克隆到本地即可
2. 然后对本地的仓库进行操作，再推送到远程即可。
## github 文件过大无法上传
https://blog.csdn.net/qq_43915356/article/details/113619750
https://deepinout.com/git/git-questions/574_git_cant_push_to_github_because_of_large_file_which_i_already_deleted.html
先删除指定文件（夹）
```shell
git filter-branch --tree-filter 'rm -rf python_data_analysis/pydata-book-3rd-edition' HEAD
```

然后强制推送
```c++
git push -f origin main
```



## 将本地仓库推送到远程
1. 首先在github中建立一个新的仓库
2. 然后根据远程仓库中
接着执行
git remote -v 可以查看远程仓库
```c++
git remote add origin git@github.com:acoding-xuan/Obsidian-file.git
git branch -M main # 这条命令将当前分支重命名为 main。
git push -u origin main # 这条命令用于将当前分支 main 的内容推送到远程仓库 origin 的 main 分支
```


或者 在本地新创建一个仓库 执行下面的操作将本地的文件推送到远程仓库

```c++
echo "# Obsidian-file" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:acoding-xuan/Obsidian-file.git
git push -u origin main
```


## 更改对应的远程仓库
如果你从别人的 GitHub 账号中克隆了一个仓库到本地，然后在本地进行修改并尝试推送到远程，默认情况下，这些更改会推送到原始克隆仓库的远程分支，也就是你克隆的那个仓库的远程分支。但是，如果你没有写权限（大多数情况下从别人的仓库克隆的情况下都是没有写权限的），那么你将无法直接推送更改到原始仓库的远程分支。此时，你有以下几种选择：

Fork 仓库：你可以先在 GitHub 上 Fork 这个仓库，创建一个属于你自己的副本。然后从你的 Fork 克隆到本地，进行修改。修改后，你可以直接推送到你自己 Fork 的仓库中。

提交 Pull Request：在你 Fork 的仓库中推送你的修改后，可以在 GitHub 上向原始仓库提交 Pull Request，请求原仓库的维护者将你的更改合并到他们的项目中。


要查看当前本地仓库的远程地址，可以使用以下命令：
git remote -v

如果需要修改远程仓库地址，可以使用：
```c++
git remote set-url origin git@github.com:acoding-xuan/SASRec.pytorch.git
git branch -M main # 这条命令将当前分支重命名为 main。
git push -u origin main # 这条命令用于将当前分支 main 的内容推送到远程仓库 origin 的 main 分支
```

总之，默认情况下，推送的目标是你最初克隆的那个仓库的远程分支，但是否成功推送取决于你是否有权限。如果没有写权限，你可以选择 Fork 仓库或更改推送目标。

### 搜索相关的技巧
找百科大全 awesome xxx  
• 找例子 xxx sample  
• 找空项目架子 xxx starter / xxx boilerplate  
• 找教程 xxx tutorial
### 找开源项目的一些途径
• https://github.com/trending/  
• https://github.com/521xueweihan/HelloGitHub  
• https://github.com/ruanyf/weekly  
• https://www.zhihu.com/column/mm-fe
### git bash 常见命令

## Git 配置
所有的配置文件，其实都保存在本地！
查看配置 git config -l
![](<assets/1699608755627.png>)

查看不同级别的配置文件：  
```
#查看系统config
git config --system --list
#查看当前用户（global）配置
git config --global  --list
```
## 设置用户名与邮箱（用户标识，必要）

当你安装 Git 后首先要做的事情是设置你的用户名称和 e-mail 地址。这是非常重要的，因为每次 Git 提交都会使用该信息。它被永远的嵌入到了你的提交中：

```
git config --global user.name "kuangshen"  #名称
git config --global user.email 24736743@qq.com   #邮箱
```

只需要做一次这个设置，如果你传递了 --global 选项，因为 Git 将总是会使用该信息来处理你在系统中所做的一切操作。如果你希望在一个特定的项目中使用不同的名称或 e-mail 地址，你可以在该项目中运行该命令而不要 --global 选项。总之 --global 为全局配置，不加为某个项目的特定配置。
![](<assets/1699608755845.png>)
## git 原理
### 三个区域
Git 本地有三个工作区域：工作目录（Working Directory）、暂存区 (Stage/Index)、资源库(Repository 或 Git Directory)。如果在加上远程的 git 仓库(Remote Directory) 就可以分为四个工作区域。文件在这四个区域之间的转换关系如下：

![](<assets/1699608755903.png>)

*   Workspace：工作区，就是你平时存放项目代码的地方
*   Index / Stage：暂存区，用于临时存放你的改动，事实上它只是一个文件，保存即将提交到文件列表信息
*   Repository：仓库区（或本地仓库），就是安全存放数据的位置，这里面有你提交到所有版本的数据。其中 HEAD 指向最新放入仓库的版本
*   Remote：远程仓库，托管代码的服务器，可以简单的认为是你项目组中的一台电脑用于远程数据交换
本地的三个区域确切的说应该是 git 仓库中 HEAD 指向的版本：
### 工作流程
git 的工作流程一般是这样的：
１、在工作目录中添加、修改文件；
２、将需要进行版本管理的文件放入暂存区域；git add . 加入暂存区
３、将暂存区域的文件提交到 git 仓库。 

因此，git 管理的文件有三种状态：已修改（modified）, 已暂存（staged）, 已提交 (committed)

![](<assets/1699608756029.png>)
![](<assets/1699608756084.png>)

### 本地仓库搭建
创建本地仓库的方法有两种：一种是创建全新的仓库，另一种是克隆远程仓库。
1、创建全新的仓库，需要用 GIT 管理的项目的根目录执行：
```
# 在当前目录新建一个Git代码库
$ git init
```
2、执行后可以看到，仅仅在项目目录多出了一个. git 目录，关于版本等的所有信息都在这个目录里面。
### 克隆远程仓库
1、另一种方式是克隆远程目录，由于是将远程服务器上的仓库完全镜像一份至本地！
```
# 克隆一个项目和它的整个代码历史(版本信息)
$ git clone [url]  # https://gitee.com/kuangstudy/openclass.git
```
### 文件的四种状态
版本控制就是对文件的版本控制，要对文件进行修改、提交等操作，首先要知道文件当前在什么状态，不然可能会提交了现在还不想提交的文件，或者要提交的文件没提交上。
*   Untracked: 未跟踪, 此文件在文件夹中, 但并没有加入到 git 库, 不参与版本控制. 通过 `git add` 状态变为 Staged.
*   Unmodify: 文件已经入库, 未修改, 即版本库中的文件快照内容与文件夹中完全一致. 这种类型的文件有两种去处, 如果它被修改, 而变为 Modified. 如果使用 `git rm` 移出版本库, 则成为 Untracked 文件
*   Modified: 文件已修改, 仅仅是修改, 并没有进行其他的操作. 这个文件也有两个去处, 通过 `git add `可进入暂存 staged 状态, 使用` git checkout` 则丢弃修改过, 返回到 unmodify 状态, 这个 git checkout 即从库中取出文件, 覆盖当前修改 !
*   Staged: 暂存状态. 执行 git commit 则将修改同步到库中, 这时库中的文件和本地文件又变为一致, 文件为 Unmodify 状态. 执行 `git reset HEAD filename` 取消暂存, 文件状态为 Modified
### 查看文件状态
上面说文件有 4 种状态，通过如下命令可以查看到文件的状态：
```
#查看指定文件状态
git status [filename]
#查看所有文件状态
git status
# git add .                  添加所有文件到暂存区
# git commit -m "消息内容"    提交暂存区中的内容到本地仓库 -m 提交信息
```
### 忽略文件
```
#为注释
*.txt        #忽略所有 .txt结尾的文件,这样的话上传就不会被选中！
!lib.txt     #但lib.txt除外
/temp        #仅忽略项目根目录下的TODO文件,不包括其它目录temp
build/       #忽略build/目录下的所有文件
doc/*.txt    #会忽略 doc/notes.txt 但不包括 doc/server/arch.txt
```


## 说明：GIT 分支
分支在 GIT 中相对较难，分支就是科幻电影里面的平行宇宙，如果两个平行宇宙互不干扰，那对现在的你也没啥影响。不过，在某个时间点，两个平行宇宙合并了，我们就需要处理一些问题了！

![](<assets/1699608756804.png>)

![](<assets/1699608756857.png>)

git 分支中常用指令：

```
# 列出所有本地分支
git branch

# 列出所有远程分支
git branch -r

# 新建一个分支，但依然停留在当前分支
git branch [branch-name]


# 新建一个分支，并切换到该分支
git checkout -b [branch]

# 合并指定分支到当前分支
$ git merge [branch]

# git status
可以查看当前分支的状态，包括对应的远程分支

# 删除分支
$ git branch -d [branch-name]

# 删除远程分支
$ git push origin --delete [branch-name]

$ git branch -dr [remote/branch]
```


## git push

###  推送到指定的远程分支
```
git push origin HEAD # 将本地当前分支的内容推送到远程仓库 origin 中与之同名的分支, 没有则会自动创建
```

```bash
git push origin HEAD:main 这条命令的作用是将本地当前分支的内容推送到远程仓库 origin 的 main 分支。具体解释如下：

git push: 用于将本地仓库的更改推送到远程仓库。
origin: 远程仓库的名称，它指向你本地 Git 仓库关联的远程仓库地址。
HEAD: 当前分支的指针，指向你当前检出的分支的最新提交。
main: 远程仓库中目标分支的名称，即你希望将本地更改推送到的远程分支。
作用
推送当前分支到远程分支: 这条命令会将你当前所在的本地分支的内容推送到远程仓库 origin 中的 main 分支。如果远程 main 分支不存在，它将被创建；如果已经存在，远程 main 分支会被更新为你当前本地分支的内容。
```
### 修改对应的远程分支
使用 -u 参数（更简洁）

```bash
git push -u origin new-branch
```
当你推送到一个新的远程分支时，使用 -u 参数会将当前分支的上游分支设置为 origin/new-branch。这条命令不仅将更改推送到远程仓库，还设置了上游分支。

## git pull

### 从远程指定分支进行拉取

可以直接进行指定
```shell
git pull <远程仓库名> <远程分支名>：<本地分支名>
eg:
git pull origin main:annotation
```

如果你想从远程的指定分支拉取更改并将其合并到本地的指定分支，可以按照以下步骤操作：

1. 切换到本地目标分支
首先，确保你在本地目标分支上。如果你还没有本地目标分支，首先需要创建并切换到它：

```bash
git checkout -b local-target-branch # 创建并切换
```

如果本地目标分支已经存在，直接切换到该分支：

```bash
git checkout local-target-branch
```

2. 拉取远程指定分支并合并到本地分支

使用以下命令从远程分支拉取更改并将其合并到当前的本地分支：

```bash
git pull origin remote-branch
```

- **`origin`**: 远程仓库的名称（通常是 `origin`）。
- **`remote-branch`**: 远程分支的名称，你希望从中拉取更改的分支。

## git reset
可以退回到之前提交的某一个版本
![](../img/Pasted%20image%2020240826190425.png)

## git连接github
```
ssh-keygen -t rsa -b 4096

回车以后要指定生成密钥的名称
```
1. 然后用文本编辑器(如notepad)打开`id_rsa.pub`这个文件, 全选复制.
2. 接下来到GitHub上，打开“Account settings”--“SSH Keys”页面，然后点“Add SSH Key”，填上Title（随意写），在Key文本框里粘贴 id_rsa.pub文件里的全部内容。
3. .验证是否成功，在git bash里输入下面的命令  
```
.验证是否成功，在git bash里输入下面的命令  
$ ssh -T git@github.com
```



## github fork 与pr

fork 是一个github 的操作能够直接复制别人的仓库（upstream repository）。

pr 可以将复制以后的仓库 提交给原来的仓库

## IDEA 中集成 Git
1、新建项目，绑定 git。

![](<assets/1699608756597.png>)

注意观察 idea 中的变化

![](<assets/1699608756672.png>)

2、修改文件，使用 IDEA 操作 git。
*   添加到暂存区
*   commit 提交
*   push 到远程仓库
IDEA 中操作  

![](<assets/1699608756916.png>)

如果同一个文件在合并分支时都被修改了则会引起冲突：解决的办法是我们可以修改冲突文件后重新提交！选择要保留他的代码还是你的代码！

master 主分支应该非常稳定，用来发布新版本，一般情况下不允许在上面工作，工作一般情况下在新建的 dev 分支上工作，工作完后，比如上要发布，或者说 dev 分支代码稳定后可以合并到主分支 master 上来。

作业练习：找一个小伙伴，一起搭建一个远程仓库，来练习 Git！

1、不要把 Git 想的很难，工作中多练习使用就自然而然的会了！

2、Git 的学习也十分多，看完我的 Git 教程之后，可以多去思考，总结到自己博客！

视频教程同步更新，请这次一定！