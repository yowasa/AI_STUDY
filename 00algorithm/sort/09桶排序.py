# 计数排序的升级版
# 将数据按范围分到n个桶中
# 顺序读取范围的桶 内部进行排序 输出
def bucket_sort_simplify(arr, max_num):
    buf = {i: [] for i in range(int(max_num) + 1)}  # 不能使用[[]]*(max+1)，这样新建的空间中各个[]是共享内存的
    arr_len = len(arr)
    for i in range(arr_len):
        num = arr[i]
        buf[int(num)].append(num)  # 将相应范围内的数据加入到[]中
    arr = []
    for i in range(len(buf)):
        if buf[i]:
            arr.extend(sorted(buf[i]))  # 这里还需要对一个范围内的数据进行排序，然后再进行输出
    return arr


lis = [3.1, 4.2, 3.3, 3.5, 2.2, 2.7, 2.9, 2.1, 1.55, 4.456, 6.12, 5.2, 5.33, 6.0, 2.12]
print(bucket_sort_simplify(lis, max(lis)))
