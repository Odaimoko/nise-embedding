# 开发过程中的碎碎念

### 并行代码思路

有三个可以并行的点

+ 多个参数
+ 多个视频
+ 多帧图

目的

+ 一个参数跑完了再跑另一个（似乎并没有必要？）——为了之后的方便。
+ 一个视频只能串行跑，但是多个视频可以同时跑。

思路

+ 先在一个参数下，跑多个视频。怎样达到？剥离每个视频的运行代码到一个函数里，多线程执行这个函数，不知道一张卡可以跑几次（依然是3次）。
+ 然后 for循环参数。

现在的问题，每个线程里的`nise_cfg`都会被重新import一次，从而参数复原。print id出来的结果是，这个`run_video`在某一个线程里，他的`nise_cfg`就用的是这个线程里的。看起来pool是把传入的参数还是什么，全部弄成pickle，然后在各自的线程里load出来。

直接向`run_video`传入`nise_cfg`和`nise_logger`可以保证参数更新。但是这样就会很快地加载flow，lock根本没有起到效果？

···是因为多个线程同时acquire了同一个锁——同样，不同的thread的lock也会新建。

```
[INFO] - Choosing GPU  1 to load flow result.
[INFO] - GPU 1's lock ACQuired. thread 140201651635968 gets lock 140199530644264
[INFO] - Choosing GPU  1 to load flow result.
[INFO] - GPU 1's lock ACQuired. thread 140343018931968 gets lock 140340898536784
```

如果只用一个GPU还好，如果用了多个GPU，就算用了multiprocessing.Lock，也会出现多个锁，从而起不到加锁的效果。

