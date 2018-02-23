#include <vector>
#include <iostream>
#include <algorithm>    // std::sort

using namespace std;

class Solution {
    public:
        vector<vector<int>> fourSum(vector<int>& nums, int target) {
            vector<vector<int>> res;
            sort(nums.begin(), nums.end());
            for (int k = 0; k < nums.size(); ++k) {
                if (k > 0 && nums[k] == nums[k - 1]) continue;
                for(int m = k + 1; m < nums.size(); ++m) {
                    if (m > 1 && nums[m] == nums[m - 1] && k != m -1) continue;
                    int temp_target = target - nums[k] - nums[m];

                    int i = m + 1, j = nums.size() - 1;
                    while (i < j) {
                        if (nums[i] + nums[j] == temp_target) {
                            res.push_back({nums[k], nums[m], nums[i], nums[j]});
                            while (i < j && nums[i] == nums[i + 1]) ++i;
                            while (i < j && nums[j] == nums[j - 1]) --j;
                            ++i;
                            --j;
                        }
                        else if (nums[i] + nums[j] < temp_target) ++i;
                        else --j;
                    }
                }
            }
            return res;

        }
};

int main() {
    vector<int> nums;
    nums.push_back(-4);
    nums.push_back(-1);
    nums.push_back(-1);
    nums.push_back(0);
    nums.push_back(1);
    nums.push_back(2);

    for(int i = 0; i < nums.size(); ++i) {
        cout << nums[i] << " ";
    }
    cout << endl;
    Solution test;

    vector<vector<int>> res = test.fourSum(nums, -1);
    for(int i = 0; i < res.size(); ++i) {
        for(int j = 0; j < res[i].size(); ++j) {
            cout << res[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
