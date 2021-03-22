
#ifndef ESPRESSO_THREAD_POOL_H
#define ESPRESSO_THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>

namespace espresso {
    class ThreadPool2 {
    public:
        typedef std::pair<std::function<void(int)>, int> TASK;

        int number() const {
            return mNumberThread;
        }
        static void enqueue(TASK&& task, int index);

        static void active();
        static void deactive();

        static int acquireWorkIndex();
        static void releaseWorkIndex(int index);

        static int init(int number);
        static void destroy();

    private:
        void enqueueInternal(TASK&& task, int index);

        static ThreadPool2* gInstance;
        ThreadPool2(int number = 0);
        ~ThreadPool2();

        std::vector<std::thread> mWorkers;
        std::vector<bool> mTaskAvailable;
        std::atomic<bool> mStop = {false};

        std::vector<std::pair<TASK, std::vector<std::atomic_bool*>>> mTasks;
        std::condition_variable mCondition;
        std::mutex mQueueMutex;

        int mNumberThread            = 0;
        std::atomic_int mActiveCount = {0};
    };
}

#endif
