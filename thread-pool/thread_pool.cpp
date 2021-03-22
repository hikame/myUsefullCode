
#include "thread_pool.h"

#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>

#define ESPRESSO_THREAD_POOL_MAX_TASKS 2

namespace espresso {
    ThreadPool2* ThreadPool2::gInstance = nullptr;
    static std::mutex gInitMutex;

    int ThreadPool2::init(int number) {
        // init保证线程池只会分配一次
        if (1 >= number){
            return 1;
        }

        std::lock_guard<std::mutex> ll(gInitMutex);
        if (nullptr != gInstance){
            if (gInstance->number() < number) {//如果设置分配的线程数number大于已经存在的线程数，维持现在的数目
                return gInstance->number();
            }
        }

        if (nullptr == gInstance){
            gInstance = new ThreadPool2(number);
        }

        return number;
    }

    void ThreadPool2::destroy() {
        std::lock_guard<std::mutex> ll(gInitMutex);
        if (nullptr != gInstance){
            delete gInstance;
            gInstance = nullptr;
        }
    }

    ThreadPool2::ThreadPool2(int numberThread){
        mNumberThread = numberThread;
        mActiveCount = 0;// 需要处理的task数
        mTaskAvailable.resize(ESPRESSO_THREAD_POOL_MAX_TASKS);
        mTasks.resize(ESPRESSO_THREAD_POOL_MAX_TASKS);

        for (int task = 0; task < mTasks.size(); ++task) {
            mTaskAvailable[task] = true;

            for (int thread_id = 0; thread_id < mNumberThread; ++thread_id) {
                mTasks[task].second.emplace_back(new std::atomic_bool(false)); //这里用来记录task是在哪个线程运行？
            }
        }
        //TODO: MNN里还有对CPU锁频时的操作
        for (int thread_id = 1; thread_id < mNumberThread; ++thread_id) {//这里为啥等于1?
            int threadIndex = thread_id;
            mWorkers.emplace_back([this, threadIndex](){
                while  (!mStop) {//线程池处于工作状态
                    while (mActiveCount > 0){
                        for (int i = 0; i < ESPRESSO_THREAD_POOL_MAX_TASKS; ++i) {
                            if (*mTasks[i].second[threadIndex]){
                                mTasks[i].first.first(threadIndex); //执行的task
                                {
                                    *mTasks[i].second[threadIndex] = false; //
                                }
                            }
                        }
                        std::this_thread::yield();//https://stackoverflow.com/questions/11048946/stdthis-threadyield-vs-stdthis-threadsleep-for
                    }
                    //当前不存在工作，阻塞当前线程，等待信号量唤醒
                    std::unique_lock<std::mutex> ll(mQueueMutex);
                    mCondition.wait(ll,[this]{ return mStop || mActiveCount > 0; });
                }
            });
        }
    }

    ThreadPool2::~ThreadPool2() {
        {
            std::lock_guard<std::mutex> ll(mQueueMutex);
            mStop = true;
        }

        mCondition.notify_all();// 唤醒所有线程，线程池要销毁了
        for (auto& worker : mWorkers){
            worker.join();
        }

        for (auto& task : mTasks){
            for(auto c : task.second){
                delete c;
            }
        }
    }

    int ThreadPool2::acquireWorkIndex() {
        if (nullptr == gInstance){
            return -1;
        }

        std::lock_guard<std::mutex> ll(gInstance->mQueueMutex);
        for (int i = 0; i < ESPRESSO_THREAD_POOL_MAX_TASKS; ++i) {
            if (gInstance->mTaskAvailable[i]){
                gInstance->mTaskAvailable[i] = false; //返回第i个task的下标，同时将其状态置为false，这个task不可用，已经在执行
                return i;
            }
        }

        return -1;
    }

    void ThreadPool2::releaseWorkIndex(int index) {
        if (nullptr == gInstance){
            return;
        }

        if (index <0 || index >= ESPRESSO_THREAD_POOL_MAX_TASKS){
            return;
        }

        std::lock_guard<std::mutex> ll(gInstance->mQueueMutex);
        gInstance->mTaskAvailable[index] = true;
    }

    void ThreadPool2::active() {
        if (nullptr == gInstance){
            return;
        }

        {
            std::lock_guard<std::mutex> ll(gInstance->mQueueMutex);
            gInstance->mActiveCount++; //活跃的task数
        }

        gInstance->mCondition.notify_all(); //通知所有线程有task了
    }

    void ThreadPool2::deactive() {
        if (nullptr == gInstance){
            return;
        }

        gInstance->mActiveCount--;
    }

    void ThreadPool2::enqueue(TASK &&task, int index) {
        if (1 >= task.second || 0 > index){ //如果task设定需要的线程数小于1或者task_index有问题，那么就直接单线程了
            for (int i = 0; i < task.second; ++i) {
                task.first(i);
            }
            return;
        }
        gInstance->enqueueInternal(std::move(task),index);
    }

    void ThreadPool2::enqueueInternal(espresso::ThreadPool2::TASK &&task, int index) {
        if (mActiveCount == 0){ //如果task为0
            for (int i = 0; i < task.second; ++i) {
                task.first(i);
            }
        }

        int workSize = task.second; // worksize是当前task指定需求的线程数
        if (workSize > mNumberThread){ // 如果指定的线程数超过了线程池的线程数，那么将task重新包装，让他循环利用所有线程
            mTasks[index].first = std::make_pair(
                        [workSize, &task, this] (int tId) {
                            for (int v = tId ; v < workSize; v+=mNumberThread) {
                                task.first(v);
                            }
                        },
                        mNumberThread
                    );
            workSize = mNumberThread; // task被重新包装了，workSize可以改成最大线程数量了
        }else{
            mTasks[index].first = std::move(task);//将task放到队列中
        }

        {
            for (int i = 1; i < workSize; ++i) {
                *mTasks[index].second[i] = true;
            }
        }

        mTasks[index].first.first(0); //主线程执行一个
        bool complete = true;

        do{
            std::this_thread::yield();
            complete = true;

            for (int i = 1; i < workSize; ++i) {//如果每个线程都把task执行完了
               if (*mTasks[index].second[i]){
                   complete = false;
                   break;
               }
            }
        } while (!complete);
    }
}


