import time
start_time=time.time()
#中间为具体程序代码
end_time=time.time()
run_time=end_time-start_time
print("程序运行时间为",run_time)
#测试level1的代码时间平均为27秒
#测试level3的代码时间平均为74秒
#虽然最终没能给出具体代码，但大致的想法是通过将图片矩阵和卷积核拉伸为一维向量，从而避免嵌套循环，实现加快计算速度