#pragma once
#include <chrono>
class AppStatistics{
    public:
        long TimeMatching = 0;
        void StartFindMatch(){
            begin = std::chrono::steady_clock::now();
        }

        void StopFindMatch(){
            auto end = std::chrono::steady_clock::now();
            TimeMatching += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        }
    private:
        std::chrono::steady_clock::time_point begin;
};