from collections import defaultdict
from math import sqrt, log10
from typing import List

'''
you are given two positive integers n and target.

An integer is considered beautiful if the sum of its digits is less than or equal to target.

Return the minimum non-negative integer x such that n + x is beautiful. The input will be generated such that it is always possible to make n beautiful.


Example 1:

Input: n = 16, target = 6
Output: 4
Explanation: Initially n is 16 and its digit sum is 1 + 6 = 7. After adding 4, n becomes 20 and digit sum becomes 2 + 0 = 2. It can be shown that we can not make n beautiful with adding non-negative integer less than 4.

nput: n = 467, target = 6
Output: 33
Explanation: Initially n is 467 and its digit sum is 4 + 6 + 7 = 17. After adding 33, n becomes 500 and digit sum becomes 5 + 0 + 0 = 5. It can be shown that we can not make n beautiful with adding non-negative integer less than 33.

'''
class Solution:
    def makeIntegerBeautiful(self, n: int, target: int) -> int:
        tail = 1
        while 1:
            # 用于进位
            m = n + (tail - n % tail) % tail
            if sum(int(i) for i in str(m)) <= target:
                return m - n
            tail *= 10









if __name__ == '__main__':
    # creators =["alice", "alice", "alice"]
    # ids=["a", "b", "c"]
    # views=[1, 2, 2]
    # # creators = ["alice", "bob"]
    # # ids = ["one", "two"]
    # # views =[1,2]
    # result = Solution().mostPopularCreator(creators, ids, views)
    # print(result)
    n = 467
    target = 6
    # need result = 33
    result = Solution().makeIntegerBeautiful(n, target)
    print(result)

    # def parse_tree_node(param):
    #     root = None
    #     nodes = []
    #     i = 0
    #     while i < len(param):
    #         if i == 0:
    #             root = TreeNode(param[i])
    #             i += 1
    #             nodes.append(root)
    #             continue
    #
    #         parent = nodes.pop(0)
    #         if param[i] is not None:
    #             left = TreeNode(param[i])
    #             parent.left = left
    #             nodes.append(left)
    #
    #         if i + 1 < len(param) and param[i + 1] is not None:
    #             right = TreeNode(param[i + 1])
    #             parent.right = right
    #             nodes.append(right)
    #
    #         i = i + 2
    #     return root
    #
    # root = parse_tree_node([5,8,9,2,1,3,7,4,6])
    # queries = [3,2,4,8]
    #
    # result = Solution().treeQueries(root, queries)
    # print(result)









