data = [[1, 4, 1], [0.5, 2, 1], [2, 2.3, 1], [1, 0.5, -1], [2, 1, -1],
        [4, 1, -1], [3.5, 4, 1], [3, 2.2, -1]]


def cal_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2
    return sum


class Node(object):
    def __init__(self, value, parent, index):
        self.value = value
        self.left = None
        self.right = None
        self.parent = parent
        self.index = index


def build(sub_data, parent_node, index, max_index):
    index += 1
    if index >= max_index:
        index = 0
    sub_data.sort(key=lambda x: x[index])
    lenght = len(sub_data)
    if lenght == 1:
        return Node(sub_data[0], parent_node, index)
    elif lenght == 2:
        node = Node(sub_data[0], parent_node, index)
        node.right = build(sub_data[1:], node, index, max_index)
        return node
    else:
        part_index = int(lenght / 2)
        node = Node(sub_data[part_index], parent_node, index)
        node.right = build(sub_data[part_index + 1:], node, index, max_index)
        node.left = build(sub_data[0: part_index], node, index, max_index)
        return node


max_index = 2
index = -1
root = build(data, None, index, max_index)

k = 3
# 找到最近节点
x = [1, 2]

cur = root
while True:
    if cur.value[cur.index] > x[cur.index]:
        if cur.left:
            cur = cur.left
        else:
            break
    elif cur.value[root.index] < x[cur.index]:
        if cur.right:
            cur = cur.right
        else:
            break
    else:
        if cur.left:
            cur = cur.left
            continue
        if cur.right:
            cur = cur.right
            continue
        else:
            break


def build_kdis(x, k_dis, cur, from_node):
    k_dis.sort(key=lambda x: x[1])
    if len(k_dis) < k:
        nest_dis = cal_distance(cur.value[0:-1], x)
        a = []
        a.append(cur.value)
        a.append(nest_dis)
        k_dis.append(a)
        if from_node == 'left' or from_node == 'parent':
            if cur.right:
                build_kdis(x, k_dis, cur.right, 'parent')
        if from_node == 'right' or from_node == 'parent':
            if cur.left:
                build_kdis(x, k_dis, cur.left, 'parent')
        if not cur.parent:
            return
        if cur == cur.parent.left:
            build_kdis(x, k_dis, cur.parent, 'left')
        else:
            build_kdis(x, k_dis, cur.parent, 'right')
    else:
        if from_node == 'parent':
            nest_dis = cal_distance(cur.value[0:-1], x)
            if k_dis[2][1] > nest_dis:
                a = []
                a.append(cur.value)
                a.append(nest_dis)
                k_dis[2] = a
            if cur.right:
                build_kdis(x, k_dis, cur.right, 'parent')
            if cur.left:
                build_kdis(x, k_dis, cur.left, 'parent')
        else:
            if abs(cur.value[cur.index] - x[cur.index]) < k_dis[k - 1][1]:
                nest_dis = cal_distance(cur.value[0:-1], x)
                if k_dis[2][1]>nest_dis:
                    a = []
                    a.append(cur.value)
                    a.append(nest_dis)
                    k_dis[2] = a
                    if from_node == 'left':
                        if cur.right:
                            build_kdis(x, k_dis, cur.right, 'parent')
                    elif from_node == 'right':
                        if cur.left:
                            build_kdis(x, k_dis, cur.left, 'parent')
            if not cur.parent:
                return
            if cur == cur.parent.left:
                build_kdis(x, k_dis, cur.parent, 'left')
            else:
                build_kdis(x, k_dis, cur.parent, 'right')


k_dis = []
build_kdis(x, k_dis, cur, 'left')

print("end")
