#ifndef __DURATION__
#define __DURATION__

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include <chrono>

using namespace std;

struct Duration
{
    public:
    Duration(double t, struct timeval s, struct timeval e)
        : time_(0.f), start_(s), end_(e)
    {
    }

    void start()
    {
        gettimeofday(&start_, NULL);
    }

    void end()
    {
        gettimeofday(&end_, NULL);
    }

    double getDuration()
    {
        time_ = end_.tv_sec - start_.tv_sec + (end_.tv_usec - start_.tv_usec) / 1000000.0;

        return time_;
    }

    private:
    double         time_;
    struct timeval start_, end_;
};

#endif // __DURATION__
