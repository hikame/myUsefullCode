#include <iostream>
#include <cstdlib>
#include <mutex>

#include "thread_pool.h"

static std::mutex gInitMutex;
static std::atomic<int> idx = {0};


#define THREAD_CONCURRENCY_BEGIN(__iter__, __num__)       \
{                                                  \
    std::pair<std::function<void(int)>, int> task; \
    task.second = __num__;                         \
    task.first  = [&](int __iter__) {

#define THREAD_CONCURRENCY_END()                                      \
    }                                                              \
    ;   \
    thread::ThreadPool2::enqueue(std::move(task), 0);\
}


namespace ThreadSafePrint
{
	static std::mutex m_CoutMutex;
	struct cout
	{
		std::unique_lock<std::mutex> m_Lock;
		cout():
			m_Lock(std::unique_lock<std::mutex>(m_CoutMutex))
		{

		}

		template<typename T>
		cout& operator<<(const T& message)
		{
			std::cout << message;
			return *this;
		}

		cout& operator<<(std::ostream& (*fp)(std::ostream&))
		{
			std::cout << fp;
			return *this;
		}
	};
}

int main(int argc, char** argv) {

    int nums = 2;

    if(argc > 1) {
        nums = std::atoi(argv[1]);
    }

    thread::ThreadPool2::init(nums);

    int task_index = thread::ThreadPool2::acquireWorkIndex();

    if (task_index >= 0){
        thread::ThreadPool2::active();
    }


    int thread_num = nums;

    THREAD_CONCURRENCY_BEGIN(tid, thread_num)
    {
        int sum = 0;
        for(int i = tid; i < 20; i+= thread_num){
            //std::lock_guard<std::mutex> ll(gInitMutex);
            //std::sync_out << "TID : " << tid << " " << i  << std::endl;
            ThreadSafePrint::cout() << "TID : " << tid << " " << i  << std::endl;
            sum += i;
        }
        ThreadSafePrint::cout() << "TID : " << tid << " SUM : " << sum  << std::endl;
    }
    THREAD_CONCURRENCY_END();


    std::cout << "nums : " << nums << std::endl;
    std::cout << "task_index : " << task_index << std::endl;

    return 0;
}
