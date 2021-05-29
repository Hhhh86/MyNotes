[TOC]



## [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

动态规划

![image-20210523170150101](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210523170150101.png?raw=true)

这是一道典型的动态规划题目，将父问题划分为一个个子问题进行解决

我们根据题目转换，可以转换成这是一题求斐波那契数列第 n 项的值的题目

斐波那契数列的定义是 f(n + 1) = f(n) + f(n - 1)*f*(*n*+1)=*f*(*n*)+*f*(*n*−1)

![image-20210523171028284](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210523171028284.png?raw=true)

**最普通的递归解法：**

```java
class Solution {
    public int numWays(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }
        return numWays(n-1)+numWays(n-2);
    }
}
```

因为该方法没有记录计算过的值，所以存在大量重复计算的操作

如numWays(4)=numWays(3)+numWays(2)

numWays(3)=numWays(2)+numWays(1)

此时，numWays(2)已经被重复计算两次

该方法耗费空间时间极大，容易超时或者栈溢出



**记忆化递归解法：**

新建一个数组存放，numWays(0)+numWays(n)的值，无需计算重复部分

我们直接在上解进行改进

```java
class Solution {
    public int numWays(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }
        int ans[] = new int[n+1];
        ans[0]=1;
        ans[1]=1;
        for(int i = 2;i<n+1;i++){
            ans[i] = (ans[i-1]+ans[i-2])%1000000007;
        }
        return ans[n];
    }
}
```

由于我们每次只需要前两个值相加，因此我们可以进一步优化空间

```java
class Solution {
    public int numWays(int n) {
        int cur = 1;
        int next = 1;
        for(int i = 0;i<n;i++){
            int temp = next;
            next = (cur + next)%1000000007;
            cur = temp;
        }
        return cur;
    }
}
```

## [ 22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

暴力破解 回溯 动态规划

![image-20210519175114090](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210519175114090.png?raw=true)

**暴力破解：**

暴力破解做这道题目难点在于要清楚构造不有效括号组合的条件。

我们可以生成所有 2^2*n* 个 `'('` 和 `')'` 字符构成的序列，然后我们检查每一个是否有效即可。

```java
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> combinations = new ArrayList<String>();
        generateAll(new char[2 * n], 0, combinations);
        return combinations;
    }

    public void generateAll(char[] current, int pos, List<String> result) {
        if (pos == current.length) {
            //符合要求才将其加入list
            if (valid(current)) {
                result.add(new String(current));
            }
        } else {
            //递归生成全部序列
            current[pos] = '(';
            generateAll(current, pos + 1, result);
            current[pos] = ')';
            generateAll(current, pos + 1, result);
        }
    }
	//判断是否符合要求的方法
    public boolean valid(char[] current) {
        int balance = 0;
        for (char c: current) {
            if (c == '(') {
                ++balance;
            } else {
                --balance;
            }
            if (balance < 0) {
                return false;
            }
        }
        return balance == 0;
    }
}
```

**回溯（最优） ：**

将问题抽象成树形状结构进行解决，采用dfs往下遍历，发现这条路不符合要求时，便返回上一节点，走另外一条路。

![image-20210519182333426](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210519182333426.png?raw=true)

这里我们为了方便理解，先将所有情况生成。

```java
class Solution {
    public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    if (n <= 0){
        return res;
    }
    dfs(n, "", res);
    return res;
}

private void dfs(int n, String path, List<String> res) {
    if (path.length() == 2 * n) {
        res.add(path);
        return;
    }

    dfs(n, path + "(", res);
    dfs(n, path + ")", res);
}
}
```

接下来进行回溯（剪枝）：

我们新建两个变量， `open` 和 `close` 分别表示左括号和右括号的个数。

组合为不有效情况为：

**当右括号个数大于左括号个数时**

**当左括号个数大于n时**

我们用上述方法判断是否回溯即可：

```java
class Solution {
    public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    if (n <= 0){
        return res;
    }
     //初始状态左右括号为0
    dfs(n, "", res,0,0);
    return res;
}
//新增参数open，close表示左右括号个数
private void dfs(int n, String path, List<String> res,int open, int close) {
    //判断是否回溯
    if(open > n || close > open){
        return;
    }
    if (path.length() == 2 * n) {
        res.add(path);
        return;
    }
	//递归时候需要更新左右括号个数
    dfs(n, path + "(", res, open+1, close);
    dfs(n, path + ")", res, open, close+1);
}
}
```

或：

```java
class Solution {
    public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    if (n <= 0){
        return res;
    }
     //初始状态左右括号为0
    dfs(n, "", res,0,0);
    return res;
}
//新增参数open，close表示左右括号个数
private void dfs(int n, String path, List<String> res，int open, int close) {
    if (path.length() == 2 * n) {
        res.add(path);
        return;
    }
    //左括号小于n时，可以继续添加左括号
	if(open < n){
       dfs(n, path + "(", res, open+1, close);
    }
    //右括号小于左括号时，可以继续添加左括号
 	if(close < open){
      dfs(n, path + ")", res, open, close+1);
    }
}
}
```

**动态规划：**

动态规划问题的两个特性是 

**（1） 最优子结构** 

**（2） 重复子问题** 

严格来说这道题目并不满足（2）这个特性，所以只能说类似是一种动态规划的解法。

举例：

比如n=1时为“（）”

那么n=2时，“0（ 1 ）2”，有0,1,2三个位置可以插入一个完整的“（）”，分别得到“（）（）”，“（（））”，“（）（）”，去除重复的就得到了n=2时的结果。

由上面的举例我们生成代码：

```java
class Solution {
    //全局list，用于判断是否重复
    List<String> all=new ArrayList<>();
    public List<String> generateParenthesis(int n) { 
        List<String> results = new ArrayList<>();
        generateParenthesisRecall(n,"",results);
        return results;
    }
    public void generateParenthesisRecall(int n,String tmp,List<String> results){
        if(n==0){
            if(tmp.length()>0){
                results.add(tmp);
            }
            return ;
        }
		//进行插入
        for(int i=0;i<tmp.length()+1;i++){
            String s=tmp.substring(0,i)+"()"+tmp.substring(i);
            //当不存在该组合时才加入
            if(!all.contains(s)) {
                all.add(s);
                generateParenthesisRecall(n - 1, s, results);
            }
        }
    } 
}
```





## [牛客.分糖果](https://www.nowcoder.com/practice/74a62e876ec341de8ab5c8662e866aef)

![image-20210524173547930](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210524173547930.png?raw=true)

因为每个小朋友需要分的糖果与隔壁两个小朋友有关，所以在遍历的时候需要关注两个的大小关系

先从左到右，再从右到左扫描两遍即可

 如：

1 2 5 4 3

往左扫描一次糖果数量为 1 2 3 1 1

往右扫描一次糖果数量为  1 2 3 2 1 

```java
import java.util.*;
public class Solution {
    public int candy(int[] ratings) {
        if(ratings == null || ratings.length <= 1){
        	return 1;
        }
        //创建存放糖果数量的数组
        int[] candy = new int[ratings.length];
        //初始化全为1
         Arrays.fill(candy, 1);
        //当前元素大于左边元素，当前元素=左边元素糖果+1
        for(int i = 1; i < candy.length; i++){
        	if(ratings[i] > ratings[i - 1])
        		candy[i] = candy[i - 1] + 1;
        }
      
        for(int i = candy.length - 1; i > 0; i--){
            //从右往左当前元素小于左边元素且当前元素糖果大于等于左边元素糖果
        	if(ratings[i] < ratings[i - 1] && candy[i] >= candy[i - 1])
                //左边元素糖果 = 当前元素糖果+1
        		candy[i - 1] = candy[i] + 1;
        }
        //统计总数
        int sum = 0;
        for(int i = 0; i < candy.length; i++)
        	sum += candy[i];
        
        return sum;
    }
}

```





##  [690. 员工的重要性](https://leetcode-cn.com/problems/employee-importance/)

递归  迭代

![image-20210523152238467](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210523152238467.png?raw=true)

写一个递归函数来统计总和，递归统计下属及其下属的重要度，最后加上自身重要度即可

**递归解法**：

```java
/*
// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};
*/
class Solution {
    public int getImportance(List<Employee> employees, int id) {
        for(Employee e : employees){
            //找到对应的员工
            if(e.id == id){
                //这个为最后一个员工
                if(e.subordinates.size() == 0){
                    return e.importance;
                }
                //若不为最后一个员工，则寻找下一个员工
                for(int userid : subordinates){
                    importance+=getImportance(employees,userid);
                }
                return e.importance;
            }
        }
        return 0;
    }
}
```

**递归解法：（优化）**

我们可以使用一个hashMap来存放list中的数据，查取的时候直接从map中取出的时间复杂度O(1)

```java
/*
// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};
*/

class Solution {
    //用map存放，每次递归时都遍历employees进行线性查找
      Map<Integer, Employee> map = new HashMap<>();
	public int getImportance(List<Employee> employees, int id) {    
        for (Employee e: employees) {
            map.put(e.id, e);
        }
        return getImportanceHelper(id);
    }
    
    public int getImportanceHelper(int id) {
        Employee employee = map.get(id);
        int ans = employee.importance;
        for (int subId: employee.subordinates) {
            ans += getImportanceHelper(subId);
        }
        return ans;
    }
  }
```

**迭代解法：**

使用队列来模拟内部栈

```java
/*
// Definition for Employee.
class Employee {
    public int id;
    public int importance;
    public List<Integer> subordinates;
};
*/

class Solution {
    public int getImportance(List<Employee> employees, int id) {
         Map<Integer, Employee> map = new HashMap<>();
        //存放数据到map
          for (Employee e: employees) {
            map.put(e.id, e);
        }
        //创建队列
        Queue<Employee>queue = new LinkedList<>();
        queue.add(map.get(id));
        int ans = 0;
        //队列不为空，取出最先放入的信息
        while(!queue.isEmpty){
          Employee e = queue.poll();
          ans+=e.importance;
            //将员工信息压入队列
            for(int userid : e.subordinates){
                queue.add(map.get(userid));
            }
        }
        return ans;
    }
}
```

# 数组专题

## [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

双指针 哈希表

![image-20210523184706563](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210523184706563.png?raw=true)

除了两个循环找到下标这种蠢方法外

我们还可以通过**双指针**的方式解决

使用双指针我们首先需要对数组进行排序，以后一个指针在数组左边，一个在右边，不断进行判断，直到找到符合条件的位置

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[]{-1, -1};
        if (nums.length == 2) {
            return new int[]{0, 1};
        }
        int[] team = Arrays.copyOf(nums, nums.length);
        Arrays.sort(team);
        int l = 0;
        int r = nums.length - 1;
        while (true) {
            if (team[l] + team[r] > target) {
                r--;
                continue;
            }
           else if (team[l] + team[r] < target) {
                l++;
                continue;
            }
            else if (team[l] + team[r] == target) {
                break;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == team[l] && result[0] == -1) {
                result[0] = i;
                continue;
            }
            if (nums[i] == team[r] && result[1] == -1) {
                result[1] = i;
            }
        }
        return result;
    }
    }
```

**哈希表（最优解法）：**

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
       Map<Integer,Integer>map = new HashMap<Integer,Integer>();
        for(int i=0;i<nums.length;i++){
            if(map.containsKey(target - nums[i])){
               return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i],i);
        }
        return new int[0];
    }
}
```



## [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

双指针

![image-20210523184320318](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210523184320318.png?raw=true)

先来蠢方法，**暴力解法**：

三重循环，解决问题

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        //hashset去重
       Set result = new HashSet();
       if(nums.length==0){
           return new ArrayList<>(result);
       }
       for(int i =0;i<nums.length;i++){
           for(int j =i+1;j<nums.length;j++){
             for(int k =j+1;k<nums.length;k++){
           if(nums[i]+nums[j]+nums[k]==0){
               List<Integer> list = new ArrayList<>();
               list.add(nums[i]);
               list.add(nums[j]);
               list.add(nums[k]);
                Collections.sort(list);
                result.add(list);
           }
       }
       }
       }
       return new ArrayList<>(result);
    }
}
```

 为了减少时间复杂度，我们可以采用两数之和的双指针思想进行解决

先对数组进行排序，固定其中的一个数，并将两个指针分别放置在左侧和右侧

如下图，固定 i 的位置，分别创建L和R在两侧作为双指针



![image-20210524165901277](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210524165901277.png?raw=true)

按照上面的方法，我们已经将三数之和问题转变成两数之和问题，只需双指针指向的值与nums[i] 刚好相加为0即可

需要注意以下条件：

因为 i 从最左侧开始固定，并且数组已经排序，则当nums[i] 大于0，则三数之和不可能为0

如果 nums[i]  == nums[i-1]，则跳过该元素nums[i] ，因为在nums[i-1]时已经将组合存入

同样，当sum == 0 时，nums[L] ==nums[L+1] 则会导致结果重复，应该跳过，L++
            当 sum == 0 时，nums[R]== nums[R-1]则会导致结果重复，应该跳过，R--

```java
class Solution {
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>>ans = new ArrayList<>();
        if(nums.length <3 || nums == null){
            return ans;
        }
        //排序
        Arrays.sort(nums);
        for(int i =0;i<nums.length;i++){
            //编写条件
            if(nums[i]>0){
                break;
            }
            if(i>0&&nums[i]==nums[i-1]){
                continue;
            }
            //设置左右边界
            int l = i+1;
            int r = nums.length-1;
            while(l<r){
                int sum = nums[i]+nums[l]+nums[r];
                //找到答案
                if(sum == 0){
                    ans.add(Arrays.asList(nums[i],nums[l],nums[r]));
                    //去重
                    while(l<r&&nums[l]==nums[l+1]){
                        l++;
                    }
                    while(l<r&&nums[r]==nums[r-1]){
                        r--;
                    }
                    l++;
                    r--;
                }
                else if(sum<0){
                    l++;
                }
                else if(sum>0){
                    r--;
                }
            }
        }
        return ans;
    }
}
```

## [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

暴力破解 双指针

![image-20210528161156566](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210528161156566.png?raw=true)

![image-20210528161212297](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210528161212297.png?raw=true)



暴力解法（超时）：

两个for循环解决，较为笨拙

```java
class Solution {
    public int maxArea(int[] height) {
        int ans = 0;
        for(int i=0;i<height.length;i++){
            for(int j=i+1;j<height.length;j++){
                ans = Math.max(Math.min(height[i],height[j])*(j-i),ans);
        }
        }
        return ans;
    }
}
```

双指针法（最优）：

由于容器的容量是由最短的板和两个板之间的距离所得，所以我们可以在数组左边右边各创建一个指针

当左边板比右边短时，记录当前容量，移动左边板，用移动后的新容量与上一个容量进行比较，谁大谁为当前容量

反之亦然

```java
class Solution {
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length-1;
        int ans = 0;
        while(left<right){
            //最短的板*底与原先的ans进行比较，谁大谁是ans
            ans = Math.max(Math.min(height[left],height[right])*(right-left),ans);
            //左边板子高度大于右边时，右边往左移动，寻找一下个有可能高过左边板子的板子
            if(height[left]>=height[right]){
                right--;
            }else{
                left++;
            }
        }
        return ans;
    }
}
```

## [283. 移动零](https://leetcode-cn.com/problems/move-zeroes/)

![image-20210528164449421](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210528164449421.png?raw=true)

**两次遍历：**

创建一个指针，用于记录非0元素个数，利用循环将非0元素全部前移，最后 j 的后面就是所需填充0的位置

![283_1.gif](https://pic.leetcode-cn.com/9669b4ffb158eaeeee6f0cd66a70f24411575edab1ab8a037c4c9084b1c743f5-283_1.gif)

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int cur = 0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=0){
                nums[cur]=nums[i];
                cur++;
            }
        }

        for(int j = cur;j<nums.length;j++){
            nums[j] =0;
        }
    }
}
```

**一次遍历：**

一次遍历的方法类似于冒泡排序，只不过是把0冒泡到后面

![283_2.gif](https://pic.leetcode-cn.com/36d1ac5d689101cbf9947465e94753c626eab7fcb736ae2175f5d87ebc85fdf0-283_2.gif)

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int cur = 0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=0){
                int temp = nums[i];
                nums[i] = nums[cur];
                nums[cur]=temp;
                cur++;
            }
        }
    }
}
```

第二种解法的本质是一个循环不变量：`在每一次循环前，j 的左边全部都是不等于0的`

- 起始`j`为0，明显满足
- 此后每一次循环中，若`nums[i] = 0`，则`j`保持不变，满足；若`nums[i] != 0`，交换后`j`增一，仍然满足

这就保证了最后结果的正确性。

## [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

双指针

![image-20210525140646843](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210525140646843.png?raw=true)

本道题目，可以合并数组后进行排序，较为简单。

**直接合并后排序:**

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
		for(int i=0;i<nums2.length;i++){
            nums1[m+i]=nums2[i];
        }
        Arrays.sort(nums1);
    }
}
```

还可以通过**双指针**进行求解：

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
      int length1 = nums1.length ;
      int length2 = nums2.length ;
        //创建一个新数组存放排序后的元素
        int arr [] = new int[length1];
        //新建数组下标
        int idx = 0;
        //两个数组下标
        int i =0,j=0;
        //因为m,n为数组有效元素个数，所以当下标小于n或者m时，表示还有元素没有进行排序
           while ( i < m || j < n) {
              //当两个下标均为超过有效元素个数时
            if (i < m && j < n) {
                //将小的元素放在arr[idx]中,idx++，并小元素所在数组下标也+1
                arr[idx++] = nums1[i] < nums2[j] ? nums1[i++] : nums2[j++];
            } //如果出现其中一个数组已经排序完，另外一个没有排序完的情况，直接将没排序完的放到arr数组
               else if (i < m) {
                arr[idx++] = nums1[i++];
            } else if (j < n) {
                arr[idx++] = nums2[j++];
            }
        }
        //将其赋值给原来数组
        for(int z =0;z<length1;z++){
            nums1[z]=arr[z];
        }
    }
}
```

由于两个数组已经有序，且nums1在数组后面保留了n个0

所以我们可以不创建数组来进行双指针

**双指针从后往前进行合并：（最优）**

![image-20210525180458335](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210525180458335.png?raw=true)

用p1和p2两个指针中元素进行比较，最大的放在p

然后p向左移 大元素的指针也左移

直至p1或者p2为0

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
 	        int p = nums1.length -1;
        	int p1 = m-1;
        	int p2 = n-1;
        while(p1>=0 || p2>=0){
           if(p1>=0&&p2>=0){
               nums1[p--]=nums1[p1] >= nums2[p2] ? nums1[p1--] : nums2[p2--];
           }else if(p1>0){
               nums1[p--]=nums1[p1--];
           }else{
               nums1[p--]=nums2[p2--];
           }
        }
    }
}
```

## [4. 寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

![image-20210525184202845](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210525184202845.png?raw=true)

![image-20210525184223892](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210525184223892.png?raw=true)

这道题目是上题的进阶挑战版

最简单粗暴的方法当然就是有序合并两个数组，然后根据元素个数奇偶关系得出中位数

**解法一：**

```java
class Solution {
public double findMedianSortedArrays(int[] nums1, int[] nums2) {    
    int m = nums1.length;
    int n = nums2.length;
    int[] nums = new int[m + n];
    //当任一数组为空，则直接从另外一个数组找中位数
    if (m == 0) {
        if (n % 2 == 0) {
            return (nums2[n / 2 - 1] + nums2[n / 2]) / 2.0;
        } else {
            return nums2[n / 2];
        }
    }
    if (n == 0) {
        if (m % 2 == 0) {
            return (nums1[m / 2 - 1] + nums1[m / 2]) / 2.0;
        } else {
            return nums1[m / 2];
        }
    }

    //新开辟数组的指针
    int count = 0;
    //两个旧数组的指针
    int i = 0, j = 0;
    //count == m+n时表示每个元素插入
    while (count != (m + n)) {
        if (i == m) {
            while (j != n) {
                nums[count++] = nums2[j++];
            }
            break;
        }
        if (j == n) {
            while (i != m) {
                nums[count++] = nums1[i++];
            }
            break;
        }

        if (nums1[i] < nums2[j]) {
            nums[count++] = nums1[i++];
        } else {
            nums[count++] = nums2[j++];
        }
    }

    if (count % 2 == 0) {
        return (nums[count / 2 - 1] + nums[count / 2]) / 2.0;
    } else {
        return nums[count / 2];
    	}
	}
}

```

# 链表专题

## [206. 反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

![image-20210526175028926](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210526175028926.png?raw=true)

![image-20210526175039208](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210526175039208.png?raw=true)

链表入门题，由于单链表只能往后面节点进行遍历，所以我们使用双指针进行反转

**迭代：**

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null){
            return head;
        }
        ListNode pre = null;
        ListNode cur = head;
        ListNode next = null; 
        while(cur!=null){
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur =next;
        }
        return pre;
    }
}
```

与双指针迭代解法不同，递归是从原链表最后一个节点开始依次向前反转

![image-20210526194102572](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210526194102572.png?raw=true)

**递归：**

```java
class Solution {
	public ListNode reverseList(ListNode head) {
         if(head == null || head.next == null){
            return head;
        }
        //依次递归找到最后一个点
        ListNode cur = reverseList(head.next);
        //此刻cur为最后一个结点，head为cur前一个节点
        //如上图，如今head为4，4.next.next = 5.next 指回4
        head.next.next = head;
        //4->5之间的关系为空,此刻5->4
        head.next = null;
        //每次递归都会返回，最终其为指向最后一个节点
        return cur;
    }
}
```



## [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

暴力破解  哈希表法  双指针

![image-20210518163605954](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210518163605954.png?raw=true)

![image-20210518163616089](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210518163616089.png?raw=true)

![image-20210518163628872](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210518163628872.png?raw=true)

这道题目除了容易想到的暴力解法O(m*n)和哈希表解法O(m+n)外，还有一种时间复杂度和哈希表一致的双指针解法。

当时面试腾讯一面的时候没答出来最后一种解法，颇为可惜。



**暴力解法（不建议）：**

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode curA = headA;
        ListNode curB = headB;
        while(curA!=null){
            while(curB!=null){
                if(curB != curA){
                    curB = curB.next;
                    
                }else{
                    return curB;
                }
            }
            curA = curA.next;
            //需要把curB重置
            curB = headB; 
        }
        return null;
    }
}

```



**下面先放哈希表的解法：**

创建一个哈希表，将任一链表的结点存进去后，再用另外一个链表的结点与哈希表里面已经存在的结点进行比较。

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        //创建哈希表, HashSet集合不允许存储相同的元素
       HashSet<ListNode>set= new HashSet<>();
        //新建一个结点当头结点
        ListNode cur = headA;
        //将headA头结点的链表放入哈希表
        while(cur!=null){
            set.add(cur);
            cur = cur.next;
        }
        
        //开始与哈希表结点进行比较
        cur = headB;
        while(cur!=null){
            if(set.contains(cur)){
                return cur;
            }
            cur = cur.next;
        }
        return null;
    }
}
//两个循环，时间复杂度是0（m+n），空间负责度为O（n）
```



**双指针解法（最优）：**

双指针无需开辟新空间，更为优秀。

创建两个指针，分别指向两个链表的头结点，然后一起往前走，当两个指针相等时，找到目标节点，直接退出。

如果其中一个指针走到最后一个位置，则将其重置为另一个链表的头指针，直至找到目标节点。

![image-20210518170353699](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210518170353699.png?raw=true)

用公式证明一下就很容易理解了：

指针A走过的结点数为：a+c+b

指针B走过的结点数为：b+c+a

当他们相交时，两个等式永远相等。



当他们不相交时：

指针A走过的结点数为：a+b

指针B走过的结点数为：b+a

一起走到最后一个结点，他们的next节点都是null，退出循环。

![image-20210518170914827](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210518170914827.png?raw=true)

```java
public class Solution {

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        //剔除特殊情况
        if (headA == null || headB == null) {
            return null;
        }
		//新建双指针
        ListNode head1 = headA;
        ListNode head2 = headB;
		
        //不相等时循环
        while (head1 != head2) {
            //当指针不为null时，往下找下一节点
            if (head1 != null) {
                head1 = head1.next;
            } else {
                head1 = headB; //为空时，跳到另一链表头结点重新开始
            }

            if (head2 != null) {
                head2 = head2.next;
            } else {
                head2 = headA;
            }
        }
        //返回结果：为相交节点或为null
        return head1;
    }
}
```

# 栈专题

##  [150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

![image-20210521211401933](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521211401933.png?raw=true)





![image-20210521211413184](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521211413184.png?raw=true)

逆波兰表达式是一种十分有用的表达式，它将复杂表达式转换为可以依靠简单的操作得到计算结果的表达式

例如(a+b)*(c+d)转换为ab+cd+*

只需两种操作（出栈，入栈）就可以完成任何普通表达式的运算

要完成这个算法，我们需要用到栈结构

栈是一个先进后出的结构，利用这个性质：

我们将数字依次压入，当遇到运算符时，依次弹出两个元素进行计算，并将计算结果重新压栈

```java
public class Solution {
    /**
     * 
     * @param tokens string字符串一维数组 
     * @return int整型
     */
    public int evalRPN (String[] tokens) {
        // write code here
        Deque<Integer>stack = new ArrayDeque<>();
        for(int i =0;i<tokens.length;i++){
           String cur = tokens[i];
            if(cur.equals("+")||cur.equals("-")||cur.equals("*")||cur.equals("/")){
                if(stack.size()<2){
                    return 0;
                }
                int after = stack.pop();
                int before = stack.pop();
                if(cur.equals("+")){
                    stack.push(before+after);
                }
                 if(cur.equals("-")){
                    stack.push(before-after);
                }
                 if(cur.equals("*")){
                    stack.push(before*after);
                }
                 if(cur.equals("/")){
                    stack.push(before/after);
                }
            }else{
                int num = Integer.parseInt(tokens[i]);
                stack.push(num);
            }
        }
        return stack.size() ==1 ? stack.pop() : 0;
    }
}
```



# 树专题

##  [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

递归  迭代

![image-20210520221005259](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520221005259.png?raw=true)

![image-20210520190822246](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520190822246.png?raw=true)

![image-20210520190837890](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520190837890.png?raw=true)

树的前中后序遍历是个重点，通常使用递归或者迭代完成。

前序遍历是按**根-左-右**的顺序依次遍历，下面列举两个不同结构的前序遍历结果。

![image-20210521171730128](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521171730128.png?raw=true)



打印顺序为：**1245367**

我们可以清晰的清楚，前序遍历就是不断的往左边找，左边没有再找右边。

先说**递归解法**：

```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        preorder(root, res);
        return res;
    }

    public void preorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        //先加入根
        res.add(root.val);
        //再加入左子节点
        preorder(root.left, res);
        //最后加入右子节点
        preorder(root.right, res);
    }
}
```

**迭代解法：**

迭代考法是面试的常客，迭代与递归类似，区别在于递归的时候隐式地维护了一个栈，而我们在迭代的时候需要显式地将这个栈模拟出来，其余的实现与细节都大致相同。

```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
  List<Integer> res = new ArrayList<>();
  //用栈来暂时存放节点
  Deque<TreeNode> stack = new ArrayDeque<>();
  //栈不为空时表示节点还可以pop，还可能找到新的右子节点
  while(root != null || !stack.isEmpty()){
    //存放当前节点信息到res，并往左子节点深入，直到节点为null
    while(root != null){
      stack.push(root);
      res.add(root.val);
      root = root.left;
    }
	//当节点的左子节点为null时，弹出节点
    root = stack.pop();
    //往右子节点继续
    root = root.right;
  }
  return res;
}
}
```

## [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

![image-20210521171755422](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521171755422.png?raw=true)

![image-20210521171825030](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521171825030.png?raw=true)

![image-20210521171810018](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521171810018.png?raw=true)

与上题前序遍历做法类似，中序遍历是依次遍历**左-根-右**

![image-20210521171730128](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521171730128.png?raw=true)

我们先找到树的最左节点，然后按顺序依次遍历。

打印顺序为：**4251627**

继续先上个**递归做法**：

先找左边，左边没有了再中间，最后才到右边

总体思路与前序遍历一致，只是插入res的先后顺序发生了改变

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        inorder(root,res);
        return res;
    }

    public void inorder(TreeNode root,List<Integer> res) {
        if(root == null){
            return;
        }
        //先找到最左边的节点
        inorder(root.left,res);
        //再添加当前节点
        res.add(root.val);
        //最后到当前节点的右子节点
        inorder(root.right,res);
    }
}
```

**迭代解法：**

```java
class Solution {
 public List<Integer> inorderTraversal(TreeNode root) {
 	List<Integer>res = new ArrayList<>();
    Deque<TreeNode>stack = new ArrayDeque<>();
     while(root!=null || !stack.isEmpty()){
         while(root!=null){
      		stack.push(root);
            root = root.left;
         }
         //当节点的左子节点为null时，弹出
         root = stack.pop();
         //res记录val
         res.add(root.val);
         //往右子节点继续
         root = root.right;
     }
     return res;
 }
 }
```

我们用一个简单的树结构来模拟一下迭代的流程：

![image-20210521174911230](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521174911230.png?raw=true)

**root!=null** 时 在栈中依次存放节点1 2

此时，因为2节点是叶子节点，所以 **root = root.left ==null **

**root ==null** 时，直接跳过while循环，来到pop()方法，将栈顶元素2节点弹出并赋为root，并记录他的val到res

 **root = root.right** 时，由于2节点没有右子节点，所以继续为null

**root ==null** 时，直接跳过while循环，来到pop()方法，将栈顶元素1节点弹出并赋为root，并记录他的val到res

 **root = root.right** 时，由于1节点有右子节点，所以root此刻为节点3

此刻，虽然stack已经空了，但是**root!=null**，所以继续进行循环

**stack.push(root)**，将节点3存入栈中，此刻栈只有节点3

因为节点3为叶子节点，所以 **root = root.left == null**

此刻**root==null**，所以不进入while循环，直接来到弹出栈顶元素节点3 **root = stack.pop()**

并将节点3的val记录到res，此刻res中的顺序为：2 1 3

接着**root = root.right == null**，此刻**root == null && stack.isEmpty()**

跳出最外层循环，返回res



## [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

![image-20210521180404501](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521180404501.png?raw=true)

![image-20210521171730128](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210521171730128.png?raw=true)

打印顺序为：4526731

**递归解法：**

和上面两题就是换了几行代码顺序

```java
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
       List<Integer>res = new ArrayList<>();
       postorder(root,res);
       return res;
    }

    public void postorder(TreeNode root,List<Integer>res){
        if(root == null){
            return;
        }
        //前中后序只需要更改下面三行代码顺序
        postorder(root.left,res);
        postorder(root.right,res);
        res.add(root.val);
    }
}
```

**迭代解法：**

需要注意：

后序遍历是依次遍历**左-右-根**，前序遍历**根-左-右**

如果把前序遍历出来的结果进行倒序，就是**右-左-根**

所以我这里稍微修改一下前序遍历的遍历顺序，将**根-左-右**改成**根-右-左**，再进行倒序即可

```java
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
  List<Integer> res = new ArrayList<>();
  //用栈来暂时存放节点
  Deque<TreeNode> stack = new ArrayDeque<>();
  //栈不为空时表示节点还可以pop，还可能找到新的右子节点
  while(root != null || !stack.isEmpty()){
    //存放当前节点信息到res，并往左子节点深入，直到节点为null
    while(root != null){
      stack.push(root);
      res.add(root.val);
      //root = root.left;
       //将根-左 改成 根-右
        root = root.right;
    }
	//当节点的右子节点为null时，弹出节点
    root = stack.pop();
    //往左子节点继续
    root = root.left;
  }
       //反转res得到左-右-根
   Collections.reverse(res);   
  return res;
}
}
```



## [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

深度遍历 层次遍历

![image-20210520164510252](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520164510252.png?raw=true)

这道题目挂个中等难度属实有点离谱。

**层次遍历：（最优）**

树的层次遍历通常采用队列来实现，将当前层的节点存放到队列中，从队列中出来的节点又将它的左右节点放入队列，直到整个树的节点被全部放入。

队列中依次弹出的节点就是我们需要的层次遍历的节点顺序。

```java
class Solution {
    public int[] levelOrder(TreeNode root) {
        if(root == null){
            return new int[0];
        }
        //创建一个队列用来保存节点
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        //存放答案的list
        ArrayList<Integer> ans = new ArrayList<>();
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            ans.add(node.val);
            if(node.left != null){
                queue.add(node.left);
            }
            if(node.right != null){
                queue.add(node.right);
            }
        }
        //将list变出数组并return
        int[] res = new int[ans.size()];
        for(int i = 0; i < ans.size(); i++)
            res[i] = ans.get(i);
        return res;
    }
}
```

**深度遍历递归：（不建议）**

```java
class Solution {
public int[] levelOrder(TreeNode root) {
    List<List<Integer>> list = new ArrayList<>();
    levelHelper(list, root, 0);
    List<Integer> tempList = new ArrayList<>();
    for (int i = 0; i < list.size(); i++) {
        tempList.addAll(list.get(i));
    }

    //把list转化为数组
    int[] res = new int[tempList.size()];
    for (int i = 0; i < tempList.size(); i++) {
        res[i] = tempList.get(i);
    }
    return res;
}

public void levelHelper(List<List<Integer>> list, TreeNode root, int height) {
    if (root == null)
        return;
    if (height >= list.size()) {
        list.add(new ArrayList<>());
    }
    list.get(height).add(root.val);
    levelHelper(list, root.left, height + 1);
    levelHelper(list, root.right, height + 1);
}
}
```

## [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

和上一题解法基本一致，只需要在加入list中改变一下代码即可。

![image-20210520172650112](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520172650112.png?raw=true)

这里直接给出**层序遍历**的代码：

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        //答案list中存放list
        ArrayList<List<Integer>>res = new ArrayList<List<Integer>>(); 
        Queue<TreeNode>queue = new LinkedList<TreeNode>();
           if(root!=null){
                queue.add(root);
           }
        
        while(!queue.isEmpty()){
             ArrayList<Integer>list = new ArrayList<Integer>(); 
            //因为是逐层节点放入list，所以需要用循环来确定list中的节点是该层节点
            for(int i=queue.size();i>0;i--){
                TreeNode node = queue.poll();
                list.add(node.val);
                if(node.left!=null){
                    queue.add(node.left);
                }
                 if(node.right!=null){
                    queue.add(node.right);
                }
            }
            res.add(list);
            
        }
        return res;
    }
}
```

## [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

事不过三，我保证这是最后一道这样的变种题了。

![image-20210520173241348](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520173241348.png?raw=true)

这题很有意思，单数层结果按正序插入队列，双数层则倒序插入。

按照这个思路，我们只需要添加一个变量来确定该层是单数层还是双数层即可。

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root != null){
            queue.add(root);
        }
        while(!queue.isEmpty()) {
            LinkedList<Integer> tmp = new LinkedList<>();
            for(int i = queue.size(); i > 0; i--) {
                TreeNode node = queue.poll();
                
                //根据res当前有多少个list来判断遍历到了第几层
                // 偶数层插入到队列头部
                if(res.size() % 2 == 0){
                    tmp.addLast(node.val);
                }
                else{
                    // 奇数层插入到队列尾部
                    tmp.addFirst(node.val);
                }  
                if(node.left != null){
                    queue.add(node.left);
                }
                if(node.right != null){
                    queue.add(node.right);
                }
            }
            res.add(tmp);
        }
        return res;
    }
}
```



##  [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

深度遍历 层次遍历

![image-20210520142856174](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520142856174.png?raw=true)

**深度遍历：**

本题目可以通过DFS来实现并找出它的最大深度，用递归实现DFS。

![image-20210520143534473](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520143534473.png?raw=true)

```java
class Solution {
    public int maxDepth(TreeNode root) {
        //如果当前节点为0，返回0
        if(root == null){
            return 0;
        }
        //将左右节点依次递归，并找出最大的节点数。
        // int left = maxDepth(root.left);
        // int right = maxDepth(root.right);
        // return Math.max(left, right) + 1;
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }
}
```

**层次遍历（BFS）：（最优）**

树的层次遍历通常采用队列来实现，将当前层的节点存放到队列中，从队列中出来的节点又将它的左右节点放入队列，直到整个树的节点被全部放入。

由于我们是获取树的最大深度，我们只需要在上述层次遍历的代码中添加一个变量用于计算遍历了多少层即可。

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null){
            return 0;
        }
        //创建一个队列用来保存节点
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        //记录每层的数量
        int res = 0;
        //当队列不为空时，表示树可能还有左右节点
        while(!queue.isEmpty()) {
            //每遍历完一层后，深度+1
          		 res++;            
            //由于queue.size()会变化，所以将其放在第一位
            //注释的循环方法i的终点会随着size的变化而变化，便不能准确把每一层的节点poll出来
            //for(int i=0;i<queue.size();++i){
            for(int i=queue.size();i>0;--i){
                //依次抛出早进入队列的节点。并判断他的左右节点是否存在，存在就加入队列
                 TreeNode node = queue.poll();
                 if(node.left!=null){
                    queue.add(node.left);
                 }
                if(node.right!=null){
                    queue.add(node.right);
                 }
            }        
        }
            return res;
    }
}
```



## [ 牛客.二叉树的最小深度](https://www.nowcoder.com/practice/e08819cfdeb34985a8de9c4e6562e724)

该题与最大深度思想大致一样，只不过需要在何时返回深度变量做点文章。

![image-20210520162619617](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210520162619617.png?raw=true)

基于二叉树的最大深度那道题目，我们同样采用层次遍历进行解决，在找到左右节点为第一个null节点时返回长度。

```java
public class Solution {
    /**
     * 
     * @param root TreeNode类 
     * @return int整型
     */
    public int run (TreeNode root) {
        // write code here
        if(root == null){
            return 0;
        }
        //当左右节点都为空时，直接返回长度1
        if(root.left==null && root.right==null){
            return 1;
        }
         //创建一个队列用来保存节点
        Queue<TreeNode> queue =  new LinkedList<>();
        queue.add(root);
        //记录深度
        int depth = 0;
        while(!queue.isEmpty()){
            //增加深度
            depth++;
            for(int i=queue.size();i>0;--i){
                TreeNode node = queue.poll();
                //找到第一个节点的左右子节点都为null时，便是最短的深度
                if(node.left==null&&node.right==null){
                    return depth;
                }
                if(node.left!=null){
                    queue.add(node.left);
                }
                if(node.right!=null){
                    queue.add(node.right);
                }
            }
        }
        return 0;
    }
}
```

# 排序

排序算法常见为十种，可分为两大类

- **比较类排序**：通过比较来决定元素间的相对次序，由于其时间复杂度不能突破O(nlogn)，因此也称为非线性时间比较类排序

- **非比较类排序**：不通过比较来决定元素间的相对次序，它可以突破基于比较排序的时间下界，以线性时间运行，因此也称为线性时间非比较类排序

  ![img](https://img2018.cnblogs.com/blog/849589/201903/849589-20190306165258970-1789860540.png?raw=true)

![img](https://images2018.cnblogs.com/blog/849589/201804/849589-20180402133438219-1946132192.png?raw=true)

## [数组排序](https://www.nowcoder.com/practice/2baf799ea0594abd974d37139de27896)

![image-20210526200638095](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210526200638095.png?raw=true)

**冒泡排序（超时）：**

冒泡排序是一种简单的排序算法。它重复地走访过要排序的数列，一次比较两个元素

如果它们的顺序错误就把它们交换过来。走访数列的工作是重复地进行直到没有再需要交换，也就是说该数列已经排序完成

这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端

![img](https://images2017.cnblogs.com/blog/849589/201710/849589-20171015223238449-2146169197.gif)

根据思想，我们用算法需要设计出：

- 比较相邻的元素。如果第一个比第二个大，就交换它们两个；
- 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对，这样在最后的元素应该会是最大的数；
- 针对所有的元素重复以上的步骤，除了最后一个；
- 重复步骤1~3，直到排序完成。

```java
import java.util.*;

public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 将给定数组排序
     * @param arr int整型一维数组 待排序的数组
     * @return int整型一维数组
     */
    public int[] MySort (int[] arr) {
        // write code here
        int length = arr.length;
        for(int i=0;i<length;++i){
            //此处-i是因为后面倒数i个元素已经有序，也可以不减
            //减1是为了防止数组越界
            for(int j=0;j<length-1-i;++j){
                if(arr[j]>arr[j+1]){
                    int temp = arr[j+1];
                    arr[j+1] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        return arr;
    }
}
```



**选择排序（超时）：**

选择排序是一种简单直观的排序算法

它的工作原理：

首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置

然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾

以此类推，直到所有元素均排序完毕

![img](https://images2017.cnblogs.com/blog/849589/201710/849589-20171015224719590-1433219824.gif)

表现最稳定的排序算法之一

因为无论什么数据进去都是O(n2)的时间复杂度，所以用到它的时候，数据规模越小越好

唯一的好处可能就是不占用额外的内存空间了吧

理论上讲，选择排序可能也是平时排序一般人想到的最多的排序方法了吧



根据思想，我们用算法需要设计出：

- 初始状态：无序区为R[1..n]，有序区为空；
- 第i趟排序(i=1,2,3…n-1)开始时，当前有序区和无序区分别为R[1..i-1]和R(i..n）。该趟排序从当前无序区中-选出关键字最小的记录 R[k]，将它与无序区的第1个记录R交换，使R[1..i]和R[i+1..n)分别变为记录个数增加1个的新有序区和记录个数减少1个的新无序区；
- n-1趟结束，数组有序化了。

```java
import java.util.*;

public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 将给定数组排序
     * @param arr int整型一维数组 待排序的数组
     * @return int整型一维数组
     */
    public int[] MySort (int[] arr) {
         int length = arr.length;
         int mincur;
        for(int i=0;i<length;++i){
            mincur = i;
            for(int j=i+1;j<length;++j){
                if(arr[mincur]>arr[j]){
                    mincur = j;
                }
            }
                int temp = arr[i];
                arr[i] = arr[mincur];
                arr[mincur] = temp;
            
        }
        return arr;
    }
}
```



**插入排序（超时）：**

插入排序的算法描述是一种简单直观的排序算法

它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入

![img](https://images2017.cnblogs.com/blog/849589/201710/849589-20171015225645277-1151100000.gif)

根据思想，我们用算法需要设计出：

- 从第一个元素开始，该元素可以认为已经被排序；

- 取出下一个元素，在已经排序的元素序列中从后向前扫描；

- 如果该元素（已排序）大于新元素，将该元素移到下一位置；

- 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；

- 将新元素插入到该位置后；

- 重复步骤2~5。

  ```java
  import java.util.*;
  
  public class Solution {
      /**
       * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
       * 将给定数组排序
       * @param arr int整型一维数组 待排序的数组
       * @return int整型一维数组
       */
      public int[] MySort (int[] arr) {
   	int length = arr.length;
      int precur,curval;
        	for(int i=0;i<length;++i){
              precur = i-1;
              curval = arr[i];
              while(precur >=0 && arr[precur]>curval){
                  arr[precur+1] = arr[precur];
                  precur--;
              }
              arr[precur+1] = curval;
          }
          return arr;
      }
  }
  ```

  

  **希尔排序（超时）：**

  简单插入排序的改进版，第一个突破O(n2)的排序算法

  它与插入排序的不同之处在于，它会优先比较距离较远的元素。希尔排序又叫 缩小增量排序

  

  原理：

  希尔排序是将待排序的数组元素按下标的一定增量分组 ，分成多个子序列

  然后对各个子序列进行直接插入排序算法排序

  然后依次缩减增量再进行排序，直到增量为1时，进行最后一次直接插入排序，排序结束

  ![img](https://upload-images.jianshu.io/upload_images/6095354-ff984d80dbc0455f.png?raw=true?raw=trueraw=trueimageMogr2/auto-orient/strip|imageView2/2/format/webp)

  根据思想，我们用算法需要设计出：

  - 分组，设定分组间隔通常为组长一半，然后根据间隔进行分组，分组之间相互排序

  - 分组之间排序通常使用插入排序

  - 分组间隔减半，继续排序，直到分组间隔为1

  - 最后，整个序列作为一个表来处理，表长度即为整个序列的长度

    

  为什么不开始就使用插入排序？

  插入排序在近乎有序的时间复杂度是最优的，因为他有一个判断条件，即当前元素大于前面元素时，就不会继续往前比较

  而希尔排序则将序列先变成近乎有序，最后再使用插入排序，就可以以近乎O（n）的时间复杂度解决排序

  数组大小越大，时间复杂度提升得越明显

```java
import java.util.*;

public class Solution {
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * 将给定数组排序
     * @param arr int整型一维数组 待排序的数组
     * @return int整型一维数组
     */
    public int[] MySort (int[] arr) {
        if(arr == null || arr.length <= 1){
        return arr;
    }
    //希尔排序  升序
    for (int d = arr.length / 2;d>0;d /= 2){ //d：增量  7   3   1
        for (int i = d; i < arr.length; i++){ 
            //i:代表即将插入的元素角标，作为每一组比较数据的最后一个元素角标 
            //j:代表与i同一组的数组元素角标
            for (int j = i-d; j>=0; j-=d){ //在此处-d 为了避免下面数组角标越界
                if (arr[j] > arr[j + d]) {// j+d 代表即将插入的元素所在的角标
                    //符合条件，插入元素（交换位置）
                    int temp = arr[j];
                    arr[j] = arr[j + d];
                    arr[j + d] = temp;
                }
            }
        } 
    }
    return arr;
    }
}
```



## [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

![image-20210526200735546](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210526200735546.png?raw=true)

![image-20210526200718259](https://github.com/Hhhh86/MyNotes/blob/master/images/image-20210526200718259.png?raw=true)