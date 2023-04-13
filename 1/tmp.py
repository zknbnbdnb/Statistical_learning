# def oddString(self, words: List[str]) -> str:
    #     # 1. 将每个字符串转换为差分数组
    #     diff = []
    #     for word in words:
    #         diff.append([ord(word[i + 1]) - ord(word[i]) for i in range(len(word) - 1)])
    #     for i in diff:
    #         if diff.count(i) == 1:
    #             return words[diff.index(i)]

    # def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
    #     res = []
    #     def one(query, word):
    #         flag = 1
    #         for i in range(len(query)):
    #             if query[i] != word[i]:
    #                 if flag:
    #                     flag = 0
    #                 else:
    #                     return False
    #         return True if flag >= 0 else False
    #     def two(query, word):
    #         flag = 2
    #         for i in range(len(query)):
    #             if query[i] != word[i]:
    #                 if flag :
    #                     flag -= 1
    #                 else:
    #                     return False
    #         return True if flag >= 0 else False
    #     for query in queries:
    #         for word in dictionary:
    #             if query in word:
    #                 res.append(word)
    #                 break
    #             if len(query) == len(word):
    #                 if one(query, word):
    #                     res.append(query)
    #                     break
    #                 elif two(query, word):
    #                     res.append(query)
    #                     break
    #     return res
# def destroyTargets(self, nums: List[int], space: int) -> int:
    #     d = defaultdict(int)
    #     for i in nums:
    #         d[i % space] += 1
    #     tmp = max(d.values())
    #     return min(i for i in nums if d[i % space] == tmp)
    # def secondGreaterElement(self, nums: List[int]) -> List[int]:
    #     zero = []
    #     one = []
    #     n = len(nums)
    #     res = [-1] * n
    #     for i in range(n):
    #         while one and one[0][0] < nums[i]:
    #             res[heappop(one)[1]] = nums[i]
    #         while zero and zero[0][0] < nums[i]:
    #             heappush(one, heappop(zero))
    #         heappush(zero, (nums[i], i))
    #     return res

if __name__ == '__main__':
    # words = ["adc","wzy","abc"]
    # result = Solution().oddString(words)
    # print(result)

    # queries =  ["yes"]
    # dictionary = ["not"]
    # result = Solution().twoEditWords(queries, dictionary)
    # print(result)

    # nums = [1,1,2,2,3,4,5,5,5,5]
    # space = 4
    # result = Solution().destroyTargets(nums, space)
    # print(result)

    nums = [1,17,18,0,18,10,20,0]
    result = Solution().secondGreaterElement(nums)
    print(result)














# def averageValue(self, nums: List[int]) -> int:
    #     res = []
    #     for num in nums:
    #         if not num % 2 and not num % 3:
    #             res.append(num)
    #     return sum(res) // len(res) if res else 0
    # def mostPopularCreator(self, creators: List[str], ids: List[str], views: List[int]) -> List[List[str]]:
    #     d = defaultdict(list)
    #     for i in range(len(creators)):
    #         d[creators[i]].append((ids[i], views[i]))
    #     tmp = max(sum(i[1] for i in d[i]) for i in d)
    #     res = []
    #     for i in d:
    #         if sum(i[1] for i in d[i]) == tmp:
    #             res.append([i, sorted(d[i], key=lambda x: (-x[1], x[0]))[0][0]])
    #     return res
    def treeQueries(self, root, queries: List[int]) -> List[int]:
        # Traverse the tree, then remove the node to find the height of the tree after removal
        height = defaultdict(int)
        def get_height(node):
            if node is None:
                return 0
            height[node] = 1 + max(get_height(node.left), get_height(node.right))
            return height[node]
        get_height(root)

        res = [0] * (len(height) + 1)
        def dfs(node, depth, res_height):
            if node is None:
                return
            depth += 1
            res[node.val] = res_height
            dfs(node.left, depth, max(res_height, depth + height[node.right]))
            dfs(node.right, depth,  max(res_height, depth + height[node.left]))
        dfs(root, -1, 0)
        for i, v in enumerate(queries):
            queries[i] = res[v]
        return queries